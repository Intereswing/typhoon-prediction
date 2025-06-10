import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from aurora import AuroraSmall, rollout
from einops import rearrange

class SinusoidPositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout_ratio=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, step=1, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embed_size, step=2, dtype=torch.float) * (-math.log(10000.0) / embed_size)
        ) # [embed_size / 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, embed_size]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)

        return x


class TTEncoder(nn.Module):
    def __init__(self, embed_size, nhead, num_layers):
        super().__init__()
        self.embed = nn.Linear(2, embed_size)
        self.pos = SinusoidPositionalEncoding(embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, nhead, batch_first=True),
            num_layers
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        x = self.transformer(x)

        return x


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim):
        super().__init__()
        self.dim = kv_dim
        self.query = nn.Linear(q_dim, self.dim)
        self.key = nn.Linear(kv_dim, self.dim)
        self.value = nn.Linear(kv_dim, self.dim)

    def forward(self, q, kv):
        Q = self.query(q) # [B, T2, D]
        K, V = self.key(kv), self.value(kv) # [B, T1, D]
        score = torch.einsum("bij,bkj->bik", Q, K) / math.sqrt(self.dim) # [B, T2, T1]
        probs = F.softmax(score, dim=-1)
        out = torch.einsum("bij,bjk->bik", probs, V)

        return out


class TTDecoder(nn.Module):
    def __init__(self, out_len, embed_size, nhead, num_layers):
        super().__init__()
        self.pos = SinusoidPositionalEncoding(embed_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, nhead, batch_first=True),
            num_layers
        )
        self.fc_out = nn.Linear(embed_size, 2)
        self.out_len = out_len

    def forward(self, x):
        traj = x[:, -1].unsqueeze(1).repeat(1, self.out_len, 1)
        traj = self.pos(traj)
        out = self.transformer(traj, x)
        out = self.fc_out(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels* 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x) # [B, 8, 240, 320] -> [B, 64, 240, 320]
        x2 = self.enc2(self.pool1(x1)) # [B, 64, 240, 320] -> [B, 64, 120, 160] -> [B, 128, 120, 160]
        x3 = self.enc3(self.pool2(x2)) # [B, 128, 120, 160] -> [B, 128, 60, 80] -> [B, 256, 60, 80]

        x4 = self.bottleneck(self.pool3(x3)) # [B, 256, 60, 80] -> [B, 256, 30, 40] -> [B, 512, 30, 40]

        y3 = self.up3(x4) # [B, 512, 30, 40] -> [B, 256, 60, 80]
        y3 = self.dec3(torch.cat([y3, x3], dim=1)) # [B, 512, 60, 80] -> [B, 256, 60, 80]

        y2 = self.up2(y3) # [B, 256, 60, 80] -> [B, 128, 120, 160]
        y2 = self.dec2(torch.cat([y2, x2], dim=1)) # [B, 256, 120, 160] -> [B, 128, 120, 160]

        y1 = self.up1(y2) # [B, 128, 120, 160] -> [B, 64, 240, 320]
        y1 = self.dec1(torch.cat([y1, x1], dim=1)) # [B, 128, 240, 320] -> [B, 64, 240, 320]

        out = self.out_conv(y1) # [B, 64, 240, 320] -> [B, 12, 240, 320]

        return out


class WeatherCNNEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # -> [B, 16, 120, 160]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # -> [B, 32, 60, 80]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> [B, 64, 30, 40]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                             # -> [B, 64, 1, 1]
        )
        self.linear = nn.Linear(64, out_dim)  # 映射到最终维度

    def forward(self, x):
        # x: [B, 12, 240, 320]
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)           # → [B*T, 1, 240, 320]
        feat = self.cnn(x)                   # → [B*T, 64, 1, 1]
        feat = feat.view(B * T, -1)          # → [B*T, 64]
        feat = self.linear(feat)             # → [B*T, out_dim]
        feat = feat.view(B, T, -1)           # → [B, 12, 64]
        return feat


class TrajectoryTransformer(nn.Module):
    def __init__(self, out_len, embed_size, nhead, num_layers, with_aurora=False, aurora_dim=512):
        super().__init__()
        self.encoder = TTEncoder(embed_size, nhead, num_layers)
        self.decoder = TTDecoder(out_len, embed_size, nhead, num_layers)
        self.with_aurora = with_aurora
        self.out_len = out_len

        if self.with_aurora:
            self.pretrained_model = AuroraSmall(autocast=True)
            aurora_checkpoint_path = "../aurora-finetune/pretrained-checkpoint/aurora-0.25-small-pretrained.ckpt"
            self.pretrained_model.load_checkpoint_local(aurora_checkpoint_path)
            self.cross_attention = Attention(aurora_dim, embed_size)
        else:
            self.cross_attention = Attention(embed_size, embed_size)

    def forward(self, x, batch=None):
        """

        :param x: [B, Lookback, 2]
        :return: [B, Horizon, 2]
        """
        enc_out = self.encoder(x) # [B, T1, embed_size]
        if self.with_aurora:
            physics_features = [phy_feat for _, phy_feat in
                                rollout(self.pretrained_model, batch, self.out_len)]
            physics_features = torch.stack(physics_features, dim=1) # [B, 12, 259200, 512]
            physics_features = physics_features.mean(-2) # [B, 12, 512]
            cross_attn_out = self.cross_attention(physics_features, enc_out)
        else:
            cross_attn_out = self.cross_attention(enc_out, enc_out)

        out = self.decoder(cross_attn_out)
        return out


class TrajectoryPredictor(nn.Module):
    def __init__(self, in_len, out_len, embed_size, nhead, num_layers):
        super().__init__()
        self.encoder = TTEncoder(embed_size, nhead, num_layers) # [B, T1, 2] -> [B, T1, D]
        self.weather_predictor = UNet(in_len, out_len, base_channels=embed_size) # [B, T1, H, W] -> [B, T2, H, W]
        self.weather_aligner = WeatherCNNEncoder(out_dim=embed_size) # [B, T2, H, W] -> [B, T2, D]
        self.cross_attn = Attention(q_dim=embed_size, kv_dim=embed_size)
        self.decoder = TTDecoder(out_len, embed_size, nhead, num_layers)

    def forward(self, obs_traj, obs_atmos):
        # obs_traj: [B, 8, 2]
        # obs_atmos: [B, 8, 240, 320]
        obs_traj_feat = self.encoder(obs_traj)
        pred_atmos = self.weather_predictor(obs_atmos)
        pred_atmos_feat = self.weather_aligner(pred_atmos)

        fuse_feature = self.cross_attn(pred_atmos_feat, obs_traj_feat)

        pred_traj = self.decoder(fuse_feature)

        return pred_traj, pred_atmos


if __name__ == "__main__":
    in_len = 8
    out_len = 12
    embed_size = 64
    nhead = 4
    num_layers = 3

    model = TrajectoryPredictor(in_len, out_len, embed_size, nhead, num_layers)
    traj = torch.randn(4, 8, 2)
    msl = torch.randn(4, 8, 240, 320)
    out_traj, out_msl = model(traj, msl)

    print(out_traj.shape)
    print(out_msl.shape)


