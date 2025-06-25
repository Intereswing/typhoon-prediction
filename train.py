import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from aurora import AuroraSmall, Tracker, rollout
from aurora.tracker import NoEyeException
from einops import rearrange, repeat, reduce

from data.dataset import TyphoonTrajectoryDataset, aurora_collate_fn
from model.typhoon_traj_model import TrajectoryTransformer, TrajectoryPredictor
from model.aurora_finetune import AuroraForTyphoon, NeuralTrackerV1, NeuralTrackerV2, NeuralTrackerV3
from utils.helper import setup_logger
from utils.metrics import haversine_torch


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


def train(local_rank, world_size):
    data_dir = "/data1/jiyilun/typhoon"
    model_name = "neural_tracker_v3"
    checkpoints_dir = Path("checkpoints") / model_name
    log_dir = Path("logs") / model_name

    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(log_dir, "train")
    batch_size = 64

    # model and optimizer
    model = nn.SyncBatchNorm.convert_sync_batchnorm(NeuralTrackerV3()).to(local_rank)
    # model.aurora.load_checkpoint_local("checkpoints/aurora/aurora-0.25-small-pretrained.ckpt")
    # for param in model.aurora.parameters():
    #     param.requires_grad = False
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
    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size / world_size), sampler=train_sampler)

    val_dataset = TyphoonTrajectoryDataset(
        data_dir, 2021, 2021, 8, 12, with_era5=True
    )
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(batch_size / world_size),
        shuffle=False,
        sampler=val_sampler,
    )

    num_epoch = 100
    step = 0
    print_per_step = 5
    num_step = num_epoch * len(train_dataloader)

    best_val_loss = float("inf")
    num_patience = 5
    patience = num_patience
    should_stop = torch.tensor([0]).to(local_rank)

    if local_rank == 0:
        logger.info(f"Train start!")

    for epoch in range(num_epoch):
        # Train
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        for obs_traj, gt_traj, pred_atmos in train_dataloader:
            step += 1

            obs_traj, gt_traj, pred_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), pred_atmos.to(local_rank)
            # obs_traj: [B, 8, 2]
            # pred_atmos: [B, 12, 1, H, W]
            obs_traj = torch.cat([obs_traj[:, -1][:, None], gt_traj[:, :-1]], dim=1) # [B, 12, 2]
            obs_traj = rearrange(obs_traj, 'b t d -> (b t) d')
            pred_atmos = rearrange(pred_atmos, 'b t c h w -> (b t) c h w')
            obs_traj_real = train_dataset.denorm_traj([obs_traj.cpu()])[0].to(local_rank)

            pred_traj = ddp_model(obs_traj, obs_traj_real, pred_atmos)
            pred_traj = rearrange(pred_traj, '(b t) d -> b t d', t=gt_traj.size(1))

            loss = loss_fn(pred_traj, gt_traj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if local_rank == 0 and step % print_per_step == 0:
                logger.info(f"[Epoch {epoch+1}/{num_epoch}] | step {step}/{num_step} | train loss: {loss.item()}")

        # Validate
        ddp_model.eval()
        total_loss = 0
        num_samples = 0
        with torch.no_grad(): # no_grad is a fn. Only after calling, it can return a context manager.
            for obs_traj, gt_traj, pred_atmos in val_dataloader:
                obs_traj, gt_traj, pred_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), pred_atmos.to(local_rank)
                hist_traj = obs_traj[:, -1]
                pred_traj_list = []
                for t in range(pred_atmos.size(1)):
                    hist_traj_real = train_dataset.denorm_traj([hist_traj.cpu()])[0].to(local_rank)
                    pred_traj = ddp_model(hist_traj, hist_traj_real, pred_atmos[:, t])
                    hist_traj = pred_traj
                    pred_traj_list.append(pred_traj)
                loss = loss_fn(rearrange(pred_traj_list, 't b d -> b t d'), gt_traj)

                total_loss += loss.item() * obs_traj.size(0)
                num_samples += obs_traj.size(0)

        # Collect validation loss.
        val_loss = torch.tensor(total_loss).to(local_rank)
        val_samples = torch.tensor(num_samples).to(local_rank)
        if local_rank == 0:
            logger.info(f"Validation loss on rank {local_rank} before all_reduce: {val_loss.item()}")
            logger.info(f"Validation samples on rank {local_rank} before all_reduce: {val_samples.item()}")

        dist.all_reduce(val_loss)
        dist.all_reduce(val_samples)

        if local_rank == 0:
            logger.info(f"Validation loss on rank {local_rank} after all_reduce: {val_loss.item()}")
            logger.info(f"Validation samples on rank {local_rank} after all_reduce: {val_samples.item()}")

        avg_val_loss = val_loss.item() / val_samples.item()

        # Save model
        if local_rank == 0:
            logger.info(f"[Epoch {epoch + 1}/{num_epoch}] | validate loss: {avg_val_loss}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = num_patience
                torch.save(
                    {k: v for k, v in ddp_model.module.state_dict().items() if not k.startswith("aurora.")},
                    checkpoints_dir / f"model_epoch_{epoch+1}.pt"
                )
                logger.info(f"Update model checkpoint in epoch {epoch+1}.")
            else:
                patience -= 1
                logger.info(f"Patience minus 1. Now {patience}.")
                if patience <= 0:
                    should_stop[0] = 1

        # Early stop.
        dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            logger.info(f"[rank {local_rank}] Early stopping triggered.")
            break

    if local_rank == 0:
        logger.info(f"Train over!")


def evaluate(local_rank):
    if local_rank == 0:
        print(f"Evaluate start!")
    data_dir = "/data1/jiyilun/typhoon"
    checkpoints_dir = Path("checkpoints") / "aurora_finetune"
    checkpoint = torch.load(checkpoints_dir / "model_epoch_9.pt")

    # model
    model = AuroraSmall().to(local_rank)
    model.load_checkpoint_local("checkpoints/aurora/aurora-0.25-small-pretrained.ckpt")
    # model.load_state_dict(checkpoint, strict=False)

    ddp_model = DDP(model, device_ids=[local_rank])

    # dataset
    test_dataset = TyphoonTrajectoryDataset(
        data_dir, 2022, 2022, 8, 12, with_era5_for_aurora=True
    )
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=test_sampler, collate_fn=aurora_collate_fn)

    # evaluate
    ddp_model.eval()
    FDE = torch.zeros(12).to(local_rank)
    num_samples = torch.tensor(0).to(local_rank)
    step = 0
    with torch.inference_mode():
        for obs_traj, gt_traj, obs_atmos in test_dataloader:
            step += 1
            obs_traj, gt_traj, obs_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), obs_atmos.to(local_rank)

            # Initialize tracker
            init = test_dataset.denorm_traj([obs_traj.cpu()])[0][0, -1]
            init = init.numpy()
            tracker = Tracker(init_lat=init[0], init_lon=init[1], init_time=obs_atmos.metadata.time[0])

            # Tracker work.
            try:
                for pred_atmos in rollout(ddp_model.module, obs_atmos, gt_traj.size(1)):
                    tracker.step(pred_atmos)
            except NoEyeException:
                print(f"Rank {local_rank} | {step}/{len(test_dataloader)} | NoEyeException detected.")
                continue

            # Retrieve results
            track = tracker.results()
            pred_traj = np.stack([track.lat[1:], track.lon[1:]], axis=1)[None] # [1, 12, 2]
            pred_traj = torch.from_numpy(pred_traj)
            gt_traj = test_dataset.denorm_traj([gt_traj.cpu()])[0] # [B, T, 2]

            FDE += torch.sum(haversine_torch(gt_traj, pred_traj), dim=0).to(local_rank) # 在batch维度上计算和。
            num_samples += gt_traj.size(0)

    dist.all_reduce(FDE)
    dist.all_reduce(num_samples)

    if local_rank == 0:
        FDE = FDE / num_samples
        print(f"Final displacement error:")
        for i in range(FDE.size(0)):
            print(f"{(i+1)*6}h: {FDE[i].item():.2f} km.")


def main():
    rank, local_rank, world_size = get_dist_info()
    ddp_setup(local_rank)
    train(local_rank, world_size)
    # evaluate(local_rank)
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

