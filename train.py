import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt

from data.dataset import TyphoonTrajectoryDataset, aurora_collate_fn
from model.model import TrajectoryTransformer, TrajectoryPredictor
from utils.helper import setup_logger

from pathlib import Path
import os


# Path and log
model_name = "Trajectory Transformer Fusing MSL with U-Net"
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/data/jiyilun/typhoon/download"
version = "1"

checkpoint_dir = Path("checkpoints") / model_name / version
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path("logs") / model_name / version
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir, "train")

# Hyperparameters
lookback = 8
horizon = 12
epochs = 100
lr = 1e-4
weight_decay = 1e-3
batch_size = 32
num_workers = 8
embed_size = 64
nhead = 4
num_layers = 3
loss_alpha = 0.1
# 1960 to 2020
train_year_start = 1960
train_year_end = 2018
val_year_start = 2019
val_year_end = 2020


# Dataloader
train_data = TyphoonTrajectoryDataset(data_dir, train_year_start, train_year_end, lookback, horizon, with_msl=True)
val_data = TyphoonTrajectoryDataset(data_dir, val_year_start, val_year_end, lookback, horizon, with_msl=True)

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True
)
val_dataloader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True
)


# Model
model = TrajectoryPredictor(lookback, horizon, embed_size, nhead, num_layers).to(device)


# Froze aurora
# for param in model.pretrained_model.parameters():
#     param.requires_grad = False


# Loss and optimizer
criterion = nn.MSELoss() # [B, out_len, 2]
optimizer = opt.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)


# Training loop
logger.info("Training start!")
best_val_loss = float("inf")
for i in range(epochs):
    # Train
    model.train()
    train_traj_loss = 0
    train_msl_loss = 0
    train_loss = 0
    for j, (x, y, msl_obs, msl_gt) in enumerate(train_dataloader):
        x, y, msl_obs, msl_gt = x.to(device), y.to(device), msl_obs.to(device), msl_gt.to(device)
        optimizer.zero_grad()

        y_p, msl_pred = model(x, msl_obs)
        # print(f"msl pred max: {torch.max(msl_pred)}")
        # print(f"msl pred min: {torch.min(msl_pred)}")
        # print("NaN in msl_gt:", torch.isnan(msl_gt).any().item())
        # print("Inf in msl_gt:", torch.isinf(msl_gt).any().item())
        # print("msl_gt max:", msl_gt.max().item())
        # print("msl_gt min:", msl_gt.min().item())
        loss_traj = criterion(y_p, y)
        loss_msl = criterion(msl_pred, msl_gt)
        loss = loss_msl * loss_alpha + loss_traj * (1 - loss_alpha)
        logger.info(f"Epoch {i+1}/{epochs} | Batch {j + 1}/{len(train_dataloader)} | "
                    f"Traj loss: {loss_traj.item()} | MSL loss: {loss_msl.item()} | Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        train_traj_loss += loss_traj.item()
        train_msl_loss += loss_msl.item()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, msl_obs, msl_gt in val_dataloader:
            x, y, msl_obs, msl_gt = x.to(device), y.to(device), msl_obs.to(device), msl_gt.to(device)

            y_p, _ = model(x, msl_obs)
            loss = criterion(y_p, y)

            # 验证集损失只关心轨迹损失
            val_loss += loss.item()

    # Message
    avg_train_traj_loss = train_traj_loss / len(train_dataloader)
    avg_train_msl_loss = train_msl_loss / len(train_dataloader)
    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)

    logger.info(f"Epoch {i+1} / {epochs} | "
                f"Train traj loss: {avg_train_traj_loss:.4f} | Train MSL loss: {avg_train_msl_loss:.4f} | "
                f"Train loss: {avg_train_loss:.4f} | "
                f"Validate loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"best_model_epoch_{i+1}.pth"
        )
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"best_model.pth"
        )
        logger.info(f"Update best_model_epoch_{i+1}.pth!")

