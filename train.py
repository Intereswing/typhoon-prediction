import datetime
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataset import TyphoonTrajectoryDataset, aurora_collate_fn
from model.model import TrajectoryTransformer, TrajectoryPredictor
from model.aurora_finetune import AuroraForTyphoon
from utils.helper import setup_logger


def get_dist_info():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return rank, local_rank, world_size


def ddp_setup(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")


def ddp_cleanup():
    dist.destroy_process_group()


def train(local_rank):
    if local_rank == 0:
        print(f"Train start!")
    data_dir = "/data/jiyilun/typhoon/download"
    checkpoints_dir = Path("checkpoints") / "aurora_finetune"
    if local_rank ==0:
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

    # model and optimizer
    model = AuroraForTyphoon(obs_len=8, pred_len=12).to(local_rank)
    model.aurora.load_checkpoint_local("checkpoints/aurora/aurora-0.25-small-pretrained.ckpt")
    for param in model.aurora.parameters():
        param.requires_grad = False
    ddp_model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, ddp_model.parameters()),
        lr=1e-4,
        weight_decay=1e-3
    )

    # dataset and sampler
    train_dataset = TyphoonTrajectoryDataset(
        data_dir, 2011, 2020, 8, 12, with_era5=True
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, collate_fn=aurora_collate_fn)

    val_dataset = TyphoonTrajectoryDataset(
        data_dir, 2021, 2021, 8, 12, with_era5=True
    )
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=val_sampler, collate_fn=aurora_collate_fn)

    num_epoch = 10
    step = 0
    print_per_step = 5
    num_step = num_epoch * len(train_dataloader)

    best_val_loss = float("inf")
    num_patience = 3
    patience = num_patience
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)

        ddp_model.train()
        for obs_traj, gt_traj, obs_atmos in train_dataloader:
            step += 1

            obs_traj, gt_traj, obs_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), obs_atmos.to(local_rank)
            pred_traj = ddp_model(obs_traj, obs_atmos)
            loss = loss_fn(pred_traj, gt_traj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if local_rank == 0 and step % print_per_step == 0:
                print(f"{datetime.datetime.now()} | [Epoch {epoch+1}/{num_epoch}] | step {step}/{num_step} | train loss: {loss.item()}")

        ddp_model.eval()
        total_loss = 0
        num_samples = 0
        with torch.no_grad(): # no_grad is a fn. Only after calling, it can return a context manager.
            for obs_traj, gt_traj, obs_atmos in val_dataloader:
                obs_traj, gt_traj, obs_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), obs_atmos.to(local_rank)
                pred_traj = ddp_model(obs_traj, obs_atmos)
                loss = loss_fn(pred_traj, gt_traj)

                total_loss += loss.item() * obs_traj.size(0)
                num_samples += obs_traj.size(0)
        val_loss = torch.tensor(total_loss).to(local_rank)
        val_samples = torch.tensor(num_samples).to(local_rank)

        dist.all_reduce(val_loss)
        dist.all_reduce(val_samples)
        avg_val_loss = val_loss.item() / val_samples.item()
        if local_rank == 0:
            print(f"[Epoch {epoch + 1}/{num_epoch}] | validate loss: {avg_val_loss}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = num_patience
                torch.save({k: v for k, v in ddp_model.module.state_dict() if not k.startswith("aurora.")}, checkpoints_dir / f"model_epoch_{epoch+1}.pt")
                print(f"Update model checkpoint in epoch {epoch+1}.")
            else:
                patience -= 1
                print(f"Patience minus 1. Now {patience}.")
                if patience <= 0:
                    break
    if local_rank == 0:
        print(f"Train over!")


def main():
    rank, local_rank, world_size = get_dist_info()
    ddp_setup(local_rank)
    train(local_rank)
    ddp_cleanup()


if __name__ == "__main__":
    main()

# # Path and log
# model_name = "Trajectory Transformer Fusing MSL with U-Net"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# data_dir = "/data/jiyilun/typhoon/download"
# version = "1"
#
# checkpoint_dir = Path("checkpoints") / model_name / version
# checkpoint_dir.mkdir(parents=True, exist_ok=True)
# log_dir = Path("logs") / model_name / version
# log_dir.mkdir(parents=True, exist_ok=True)
# logger = setup_logger(log_dir, "train")
#
# # Training loop
# logger.info("Training start!")
# best_val_loss = float("inf")
# for i in range(epochs):
#     # Train
#     model.train()
#     train_traj_loss = 0
#     train_msl_loss = 0
#     train_loss = 0
#     for j, (x, y, msl_obs, msl_gt) in enumerate(train_dataloader):
#         x, y, msl_obs, msl_gt = x.to(device), y.to(device), msl_obs.to(device), msl_gt.to(device)
#         optimizer.zero_grad()
#
#         y_p, msl_pred = model(x, msl_obs)
#         # print(f"msl pred max: {torch.max(msl_pred)}")
#         # print(f"msl pred min: {torch.min(msl_pred)}")
#         # print("NaN in msl_gt:", torch.isnan(msl_gt).any().item())
#         # print("Inf in msl_gt:", torch.isinf(msl_gt).any().item())
#         # print("msl_gt max:", msl_gt.max().item())
#         # print("msl_gt min:", msl_gt.min().item())
#         loss_traj = criterion(y_p, y)
#         loss_msl = criterion(msl_pred, msl_gt)
#         loss = loss_msl * loss_alpha + loss_traj * (1 - loss_alpha)
#         logger.info(f"Epoch {i+1}/{epochs} | Batch {j + 1}/{len(train_dataloader)} | "
#                     f"Traj loss: {loss_traj.item()} | MSL loss: {loss_msl.item()} | Loss: {loss.item()}")
#
#         loss.backward()
#         optimizer.step()
#         train_traj_loss += loss_traj.item()
#         train_msl_loss += loss_msl.item()
#         train_loss += loss.item()
#
#     # Validate
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for x, y, msl_obs, msl_gt in val_dataloader:
#             x, y, msl_obs, msl_gt = x.to(device), y.to(device), msl_obs.to(device), msl_gt.to(device)
#
#             y_p, _ = model(x, msl_obs)
#             loss = criterion(y_p, y)
#
#             # 验证集损失只关心轨迹损失
#             val_loss += loss.item()
#
#     # Message
#     avg_train_traj_loss = train_traj_loss / len(train_dataloader)
#     avg_train_msl_loss = train_msl_loss / len(train_dataloader)
#     avg_train_loss = train_loss / len(train_dataloader)
#     avg_val_loss = val_loss / len(val_dataloader)
#
#     logger.info(f"Epoch {i+1} / {epochs} | "
#                 f"Train traj loss: {avg_train_traj_loss:.4f} | Train MSL loss: {avg_train_msl_loss:.4f} | "
#                 f"Train loss: {avg_train_loss:.4f} | "
#                 f"Validate loss: {avg_val_loss:.4f}")
#
#     # Save best model
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(
#             model.state_dict(),
#             checkpoint_dir / f"best_model_epoch_{i+1}.pth"
#         )
#         torch.save(
#             model.state_dict(),
#             checkpoint_dir / f"best_model.pth"
#         )
#         logger.info(f"Update best_model_epoch_{i+1}.pth!")

