from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from aurora import AuroraSmall, rollout, Batch, Metadata
from sqlalchemy.engine import cursor

from model.layers.FPN import FPN, Bottleneck

def crop_atmos(coords, atmos, delta=5.0):
    B, C, H, W = atmos.shape
    device = atmos.device

    lat_res = 0.25
    lon_res = 0.25
    crop_size = int(2 * delta / lat_res)

    lat_vals = torch.linspace(90, -90, H, device=device)
    lon_vals = torch.linspace(0, 360 - lon_res, W, device=device)

    lat_centers = coords[:, 0]
    lon_centers = coords[:, 1]

    lat_idx = torch.argmin(torch.abs(lat_vals[None, :] - lat_centers[:, None]), dim=1) # [B]
    lon_idx = torch.argmin(torch.abs(lon_vals[None, :] - lon_centers[:, None]), dim=1) # [B]

    offset = crop_size // 2

    lat_indices = lat_idx[:, None] + torch.arange(-offset, -offset + crop_size, device=device) # [B, 40]
    lon_indices = lon_idx[:, None] + torch.arange(-offset, -offset + crop_size, device=device) # [B, 40]

    lat_idx_grid = lat_indices[:, None, :, None].expand(-1, C, -1, W) # [B, C, H_c, W]
    lon_idx_grid = lon_indices[:, None, None, :].expand(-1, C, crop_size, -1) # [B, C, H_c, W_c]

    cropped_atmos = torch.gather(atmos, dim=2, index=lat_idx_grid)
    cropped_atmos = torch.gather(cropped_atmos, dim=3, index=lon_idx_grid)
    return cropped_atmos


def linear_regression(traj):
    b, t, d = traj.shape
    device = traj.device

    time_steps = torch.arange(t, dtype=torch.float, device=device)
    time_steps = rearrange(time_steps, 't -> 1 t 1')
    ones = torch.ones_like(time_steps)
    X = torch.cat([time_steps, ones], dim=-1) # 1 t 2

    Xt = rearrange(X, 'b t d -> b d t')
    XtX = Xt @ X
    XtX_inv = torch.inverse(XtX)
    X_pseudo_inv = XtX_inv @ Xt

    coefficients = X_pseudo_inv @ traj
    a, b = coefficients[:, 0, :], coefficients[:, 1, :]

    t_next = t
    pred = a * t_next + b

    return pred


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2), # [B, 32, 240, 320]
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 120, 160]
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 60, 80]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # [B, 128, 1, 1]
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.encoder(x).view(B, T, -1)
        x = self.fc(x)
        return x


class AuroraForTyphoon(nn.Module):
    def __init__(self, obs_len, pred_len, cnn_hidden=128, rnn_hidden=64):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.aurora = AuroraSmall()
        self.cnn_encoder = CNNEncoder(in_channels=2, hidden_dim=cnn_hidden)
        self.rnn_encoder = nn.GRU(input_size=2, hidden_size=rnn_hidden, batch_first=True)
        self.rnn_decoder = nn.GRU(input_size=cnn_hidden + rnn_hidden, hidden_size=rnn_hidden, batch_first=True)
        self.fc_out = nn.Linear(rnn_hidden, 2)

    def forward(self, obs_traj, obs_atmos):
        # obs_traj: [B, T, 2]
        # obs_atmos: [B, 2, C, H, W]
        pred_atmos_select_local_list = []
        for pred_atmos in rollout(self.aurora, obs_atmos, self.pred_len):
            pred_atmos = pred_atmos.normalise(dict())

            z700_index = list(pred_atmos.metadata.atmos_levels).index(700)
            lat60_index = list(pred_atmos.metadata.lat).index(60)
            lat0_index = list(pred_atmos.metadata.lat).index(0)
            lon100_index = list(pred_atmos.metadata.lon).index(100)
            lon180_index = list(pred_atmos.metadata.lon).index(180)

            pred_msl = pred_atmos.surf_vars["msl"] # [B, 1, H, W]
            pred_msl_local = pred_msl[:, :, lat60_index + 1: lat0_index + 1, lon100_index: lon180_index]
            pred_z700 = pred_atmos.atmos_vars["z"][:, :, z700_index] # [B, 1, H, W]
            pred_z700_local = pred_z700[:, :, lat60_index + 1: lat0_index + 1, lon100_index: lon180_index]
            pred_atmos_select_local_list.append(torch.stack([pred_msl_local, pred_z700_local], dim=2))
        pred_atmos_select_local = torch.cat(pred_atmos_select_local_list, dim=1) # [B, T', 2, H, W]

        pred_atmos_feat = self.cnn_encoder(pred_atmos_select_local) # [B, T', 128]
        _, h_obs_traj = self.rnn_encoder(obs_traj) # [1, B, 64]
        h_obs_traj = h_obs_traj.permute(1, 0, 2).repeat(1, self.pred_len, 1) # [B, T', 64]
        rnn_out, _ = self.rnn_decoder(torch.cat([pred_atmos_feat, h_obs_traj], dim=-1)) # [B, T', 64]
        out = self.fc_out(rnn_out)
        return out


'''Past trajectory and future local graph -> future trajectory'''
class NeuralTrackerV1(nn.Module):
    def __init__(self, obs_len, pred_len, hidden_size=64):
        super().__init__()
        self.embedding = nn.Linear(2, hidden_size)
        self.traj_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=3,
        )
        self.atmos_encoder = CNNEncoder(in_channels=1, hidden_dim=hidden_size)
        self.traj_decoder = nn.LSTM(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=3,
        )
        self.out_net = nn.Linear(hidden_size, 2)
        self.obs_len = obs_len
        self.pred_len = pred_len

    def crop(self, atmos):
        # atmos: b t c h w
        device = atmos.device
        lats = torch.linspace(90, -90, 721).to(device)
        lons = torch.linspace(0, 360, 1440 + 1)[:-1].to(device)
        lat_indices = (lats < 60) & (lats >= 0)
        lon_indices = (lons < 180) & (lons >= 100)
        crop_atmos = atmos[..., lat_indices, :][..., lon_indices]

        return crop_atmos

    def forward(self, obs_traj, pred_atmos):
        # obs_traj: [b, 8, 2]
        # pred_atmos: [b, 12, c, 721, 1440]
        obs_traj_embed = self.embedding(obs_traj) # [B, T, 64]
        _, (obs_traj_h, _) = self.traj_encoder(obs_traj_embed) # [3, B, 64]
        obs_traj_h = obs_traj_h[-1] # [B, 64]
        obs_traj_h = repeat(obs_traj_h, 'b h -> b t h', t=self.pred_len)  # [B, 12, 64]
        pred_atmos_h = self.atmos_encoder(self.crop(pred_atmos))  # [B, 12, 64]
        fused_h = torch.cat([obs_traj_h, pred_atmos_h], dim=-1)  # [B, 12, 128]
        pred_traj_h, (_, _) = self.traj_decoder(fused_h) # [B, 12, 64]
        pred_traj = self.out_net(pred_traj_h) # [B, 12, 2]
        return pred_traj


'''One coordinate and one global graph -> next coordinate'''
class NeuralTrackerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.traj_encoder = nn.Linear(2, 64)
        self.atmos_encoder = FPN(block=Bottleneck, num_blocks=[2, 2, 2, 2], in_channels=2)
        self.traj_decoder = nn.Sequential(
            nn.Linear(64 + 256*4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )


    def forward(self, traj, pred_atmos):
        # traj: [B, 2]
        # pred_atmos: [B, C, H, W]
        traj_embed = self.traj_encoder(traj)
        p2, p3, p4, p5 = self.atmos_encoder(pred_atmos)
        weather_feat = []
        for p in [p2, p3, p4, p5]:
            pooled = F.adaptive_avg_pool2d(p, output_size=1)  # shape: [B, 256, 1, 1]
            weather_feat.append(pooled.squeeze(-1).squeeze(-1))  # [B, 256]
        weather_embed = torch.cat(weather_feat, dim=-1)  # shape: [B, 256*4] = [B, 1024]
        out = self.traj_decoder(torch.cat([weather_embed, traj_embed], dim=-1))

        return out


'''One coordinate and one surrounding graph -> next coordinate'''
class NeuralTrackerV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.traj_encoder = nn.Linear(2, 64)
        self.atmos_encoder = FPN(block=Bottleneck, num_blocks=[2, 2, 2, 2], in_channels=2)
        self.traj_decoder = nn.Sequential(
            nn.Linear(64 + 256 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, traj, traj_real, pred_atmos):
        # traj: [B, 2]
        # pred_atmos: [B, C, H, W]
        traj_embed = self.traj_encoder(traj)

        pred_atmos_crop = crop_atmos(traj_real, pred_atmos, delta=5.0) # b c 40 40
        p2, p3, p4, p5 = self.atmos_encoder(pred_atmos_crop)
        weather_feat = []
        for p in [p2, p3, p4, p5]:
            pooled = F.adaptive_avg_pool2d(p, output_size=1)  # shape: [B, 256, 1, 1]
            weather_feat.append(pooled.squeeze(-1).squeeze(-1))  # [B, 256]
        weather_embed = torch.cat(weather_feat, dim=-1)  # shape: [B, 256*4] = [B, 1024]

        out = self.traj_decoder(torch.cat([weather_embed, traj_embed], dim=-1))

        return out


'''One linear extrapolated coordinate and one surrounding graph -> next coordinate'''
# coordinate decoder: 1-layer MLP
class NeuralTrackerV4(nn.Module):
    def __init__(self, traj_mean, traj_std):
        super().__init__()
        self.coord_encoder = nn.Linear(2, 64)
        self.atmos_encoder = FPN(block=Bottleneck, num_blocks=[2, 2, 2, 2], in_channels=2)
        self.coord_decoder = nn.Linear(64 + 256 * 4, 2)
        self.register_buffer('traj_mean', traj_mean)
        self.register_buffer('traj_std', traj_std)

    def forward(self, obs_traj, pred_atmos):
        # obs_traj_real: b 8 2
        # pred_atmos: b c h w
        obs_traj_real = obs_traj * self.traj_std + self.traj_mean
        pred_coord_real = linear_regression(obs_traj_real)
        pred_coord = (pred_coord_real - self.traj_mean) / self.traj_std
        pred_atmos_crop = crop_atmos(pred_coord_real, pred_atmos, delta=5.0)

        coord_embed = self.coord_encoder(pred_coord)
        p2, p3, p4, p5 = self.atmos_encoder(pred_atmos_crop)
        weather_feat = []
        for p in [p2, p3, p4, p5]:
            pooled = F.adaptive_avg_pool2d(p, output_size=1)  # shape: [B, 256, 1, 1]
            weather_feat.append(pooled.squeeze(-1).squeeze(-1))  # [B, 256]
        weather_embed = torch.cat(weather_feat, dim=-1)  # shape: [B, 256*4] = [B, 1024]
        out = self.coord_decoder(torch.cat([weather_embed, coord_embed], dim=-1))

        return out


# coordinate decoder: 2-layer MLP
class NeuralTrackerV5(nn.Module):
    def __init__(self, traj_mean, traj_std):
        super().__init__()
        self.coord_encoder = nn.Linear(2, 64)
        self.atmos_encoder = FPN(block=Bottleneck, num_blocks=[2, 2, 2, 2], in_channels=2)
        self.coord_decoder = nn.Sequential(
            nn.Linear(64 + 256 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.register_buffer('traj_mean', traj_mean)
        self.register_buffer('traj_std', traj_std)

    def forward(self, obs_traj, pred_atmos):
        # obs_traj_real: b 8 2
        # pred_atmos: b c h w
        obs_traj_real = obs_traj * self.traj_std + self.traj_mean
        pred_coord_real = linear_regression(obs_traj_real)
        pred_coord = (pred_coord_real - self.traj_mean) / self.traj_std
        pred_atmos_crop = crop_atmos(pred_coord_real, pred_atmos, delta=5.0)

        coord_embed = self.coord_encoder(pred_coord)
        p2, p3, p4, p5 = self.atmos_encoder(pred_atmos_crop)
        weather_feat = []
        for p in [p2, p3, p4, p5]:
            pooled = F.adaptive_avg_pool2d(p, output_size=1)  # shape: [B, 256, 1, 1]
            weather_feat.append(pooled.squeeze(-1).squeeze(-1))  # [B, 256]
        weather_embed = torch.cat(weather_feat, dim=-1)  # shape: [B, 256*4] = [B, 1024]
        out = self.coord_decoder(torch.cat([weather_embed, coord_embed], dim=-1))

        return out



if __name__ == "__main__":
    traj = torch.stack([torch.arange(3), torch.arange(3)], dim=-1)[None].float()
    print(traj.shape)
    print(linear_regression(traj).shape)
    # batch = Batch(
    #     surf_vars={k: torch.randn(1, 2, 721, 1440) for k in ("2t", "10u", "10v", "msl")},
    #     static_vars={k: torch.randn(721, 1440) for k in ("lsm", "z", "slt")},
    #     atmos_vars={k: torch.randn(1, 2, 13, 721, 1440) for k in ("z", "u", "v", "t", "q")},
    #     metadata=Metadata(
    #         lat=torch.linspace(90, -90, 721),
    #         lon=torch.linspace(0, 360, 1440 + 1)[:-1],
    #         time=(datetime(2020, 6, 1, 12, 0),),
    #         atmos_levels=(50, 100, 150,200,250,300,400,500,600,700,850,925,1000),
    #     ),
    # )
    # p_atmos = torch.randn(1, 1, 721, 1440)
    # hist_traj = torch.randn(1, 2)
    # hist_traj_real = torch.tensor([[30, 140]])
    #
    # model = NeuralTrackerV3()
    #
    # t0 = datetime.now()
    # pred = model.forward(hist_traj, hist_traj_real, p_atmos)
    # t1 = datetime.now()
    # print(f"Calculate time: {(t1 - t0).total_seconds():.2f}s")
    # print(pred.size())
