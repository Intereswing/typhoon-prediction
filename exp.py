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
from model.aurora_finetune import AuroraForTyphoon, NeuralTrackerV1, NeuralTrackerV2, NeuralTrackerV3, NeuralTrackerV4, \
    NeuralTrackerV5
from utils.helper import setup_logger
from utils.metrics import haversine_torch
from utils.visualization import plot_and_save


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
    model_name = "neural_tracker_v5"
    checkpoints_dir = Path("checkpoints") / model_name
    log_dir = Path("logs") / model_name

    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(log_dir, "train")
    batch_size = 64

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

    # model and optimizer
    model = NeuralTrackerV5(
        traj_mean=train_dataset.mean,
        traj_std=train_dataset.std,
    ).to(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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


    num_epoch = 100
    step = 0
    print_per_step = 5
    num_step = num_epoch * len(train_dataloader)

    best_val_loss = float("inf")
    num_patience = 10
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

            # obs_traj: [B, 8, 2]
            # pred_atmos: [B, 12, 1, H, W]
            obs_traj, gt_traj, pred_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), pred_atmos.to(local_rank)

            '''auto-regressive teacher force training.'''
            full_traj = torch.cat([obs_traj, gt_traj], dim=1)
            obs_traj_train = [full_traj[:, i: (i+8), :] for i in range(gt_traj.shape[1])]
            obs_traj_train = rearrange(obs_traj_train, 't2 b t1 d -> (b t2) t1 d')
            pred_atmos = rearrange(pred_atmos, 'b t2 c h w -> (b t2) c h w')

            pred_traj = ddp_model(obs_traj_train, pred_atmos)
            pred_traj = rearrange(pred_traj, '(b t2) d -> b t2 d', t2=gt_traj.size(1))

            '''end to end training'''
            # pred_traj = ddp_model(obs_traj, pred_atmos)

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
        with torch.no_grad(): # No_grad is a fn. Only after calling, it can return a context manager.
            for obs_traj, gt_traj, pred_atmos in val_dataloader:
                obs_traj, gt_traj, pred_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), pred_atmos.to(local_rank)

                '''auto-regressive inference'''
                hist_traj = obs_traj
                pred_traj = []
                for t in range(gt_traj.shape[1]):
                    pred_coord = ddp_model(hist_traj, pred_atmos[:, t])
                    hist_traj = torch.cat([hist_traj[:, 1:], pred_coord[:, None]], dim=1)
                    pred_traj.append(pred_coord)
                pred_traj = rearrange(pred_traj, 't2 b d -> b t2 d')

                '''end to end inference'''
                # pred_traj = ddp_model(obs_traj, pred_atmos)

                '''calculate loss'''
                loss = loss_fn(pred_traj, gt_traj)

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
    model_name = "neural_tracker_v5"
    checkpoints_dir = Path("checkpoints") / model_name
    checkpoint = torch.load(checkpoints_dir / "model_epoch_62.pt")

    log_dir = Path("logs") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir, 'evaluate')

    fig_dir = Path("figures") / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    test_dataset = TyphoonTrajectoryDataset(
        data_dir, 2022, 2022, 8, 12, with_era5=True
    )
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, sampler=test_sampler)

    # model
    model = NeuralTrackerV5(
        traj_mean=test_dataset.mean,
        traj_std=test_dataset.std,
    ).to(local_rank)
    model.load_state_dict(checkpoint, strict=True)
    ddp_model = DDP(model, device_ids=[local_rank])


    # evaluate
    ddp_model.eval()
    FDE = torch.zeros(12).to(local_rank)
    num_samples = torch.tensor(0).to(local_rank)
    step = 0
    with torch.inference_mode():
        for obs_traj, gt_traj, pred_atmos in test_dataloader:
            step += 1
            obs_traj, gt_traj, pred_atmos = obs_traj.to(local_rank), gt_traj.to(local_rank), pred_atmos.to(local_rank)

            '''aurora tracker inference'''
            # init = test_dataset.denorm_traj([obs_traj.cpu()])[0][0, -1]
            # init = init.numpy()
            # tracker = Tracker(init_lat=init[0], init_lon=init[1], init_time=obs_atmos.metadata.time[0])
            # try:
            #     for pred_atmos in rollout(ddp_model.module, obs_atmos, gt_traj.size(1)):
            #         tracker.step(pred_atmos)
            # except NoEyeException:
            #     print(f"Rank {local_rank} | {step}/{len(test_dataloader)} | NoEyeException detected.")
            #     continue
            # track = tracker.results()
            # pred_traj = np.stack([track.lat[1:], track.lon[1:]], axis=1)[None] # [1, 12, 2]
            # pred_traj = torch.from_numpy(pred_traj)

            '''end to end inference'''
            # pred_traj = ddp_model(obs_traj, pred_atmos)
            # pred_traj = test_dataset.denorm_traj(pred_traj.cpu())

            '''auto-regressive inference'''
            hist_traj = obs_traj
            pred_traj = []
            for t in range(gt_traj.shape[1]):
                pred_coord = ddp_model(hist_traj, pred_atmos[:, t])
                hist_traj = torch.cat([hist_traj[:, 1:], pred_coord[:, None]], dim=1)
                pred_traj.append(pred_coord)
            pred_traj = rearrange(pred_traj, 't2 b d -> b t2 d')

            pred_traj = test_dataset.denorm_traj(pred_traj.cpu())

            gt_traj = test_dataset.denorm_traj(gt_traj.cpu()) # [B, T, 2]

            FDE += torch.sum(haversine_torch(gt_traj, pred_traj), dim=0).to(local_rank) # 在batch维度上计算和。
            num_samples += gt_traj.size(0)

    dist.all_reduce(FDE)
    dist.all_reduce(num_samples)

    if local_rank == 0:
        FDE = FDE / num_samples
        logger.info(f"Final displacement error:")
        for i in range(FDE.size(0)):
            logger.info(f"{(i+1)*6}h: {FDE[i].item():.2f} km.")

        # with torch.inference_mode():
        #     visualize_prediction(ddp_model, test_dataset, fig_dir, local_rank)


def visualize_prediction(model, dataset, fig_dir, device):
    for i in [0, 20, 43, 60, 80, 100]:
        obs_traj, gt_traj, pred_atmos = dataset[i]
        obs_traj, gt_traj, pred_atmos = obs_traj[None].to(device), gt_traj[None].to(device), pred_atmos[None].to(device)

        '''end to end inference'''
        # pred_traj = model(obs_traj, pred_atmos)
        # pred_traj = pred_traj.squeeze(0)

        '''auto-regressive inference'''
        hist_traj = obs_traj
        pred_traj = []
        for t in range(gt_traj.shape[1]):
            pred_coord = model(hist_traj, pred_atmos[:, t])
            hist_traj = torch.cat([hist_traj[:, 1:], pred_coord[:, None]], dim=1)
            pred_traj.append(pred_coord)
        pred_traj = rearrange(pred_traj, 't2 b d -> b t2 d')

        obs_traj, gt_traj, pred_traj = obs_traj.squeeze(0), gt_traj.squeeze(0), pred_traj.squeeze(0)
        obs_traj, gt_traj, pred_traj = dataset.denorm_traj([obs_traj.cpu(), gt_traj.cpu(), pred_traj.cpu()])
        plot_and_save(obs_traj, gt_traj, pred_traj, i, dataset, fig_dir)


def main():
    rank, local_rank, world_size = get_dist_info()
    ddp_setup(local_rank)
    # train(local_rank, world_size)
    evaluate(local_rank)
    ddp_cleanup()


if __name__ == "__main__":
    main()

