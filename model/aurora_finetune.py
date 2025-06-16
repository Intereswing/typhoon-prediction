from datetime import datetime

import torch
import torch.nn as nn
from aurora import AuroraSmall, rollout, Batch, Metadata


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2), # [B, 32, 240, 320]
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 120, 160]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, hidden_dim)

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


if __name__ == "__main__":
    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 721, 1440) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(721, 1440) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 13, 721, 1440) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 721),
            lon=torch.linspace(0, 360, 1440 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(50, 100, 150,200,250,300,400,500,600,700,850,925,1000),
        ),
    ).to("cuda")
    hist_traj = torch.randn(1, 8, 2).to("cuda")

    model = AuroraForTyphoon(8, 12).to("cuda")
    model.train()
    for param in model.aurora.parameters():
        param.requires_grad = False

    t0 = datetime.now()
    pred = model.forward(hist_traj, batch)
    t1 = datetime.now()
    print(f"Calculate time: {(t1 - t0).total_seconds():.2f}s")
    print(pred.size())
