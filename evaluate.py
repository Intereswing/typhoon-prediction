import numpy as np
import torch
import torch.nn as nn
from aurora import Aurora, rollout, Tracker
from aurora.tracker import NoEyeException
from torch.utils.data import DataLoader
import torch.optim as opt

from data.dataset import TyphoonTrajectoryDataset, aurora_collate_fn
from model.model import TrajectoryTransformer, TrajectoryPredictor
from utils.helper import setup_logger
from utils.metrics import haversine
from utils.visualization import plot_and_save

from pathlib import Path
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/data/jiyilun/typhoon/download"

# Change to your model.
model_name = "Trajectory Transformer Fusing MSL with U-Net"
version = "1"

checkpoint_dir = Path("checkpoints") / model_name / version
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoint_dir / "best_model_epoch_27.pth"

fig_dir = Path("figs") / model_name / version
fig_dir.mkdir(parents=True, exist_ok=True)

log_dir = Path("logs") / model_name / version
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir, "evaluate")


lookback = 8
horizon = 12
batch_size = 32
num_workers = 8
embed_size = 64
nhead = 4
num_layers = 3

test_data = TyphoonTrajectoryDataset(data_dir, 2022, 2022, lookback, horizon, with_msl=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=True)

model = TrajectoryPredictor(lookback, horizon, embed_size, nhead, num_layers).to(device)
model.load_state_dict(torch.load(checkpoint_file), strict=True)
model.eval()

with torch.inference_mode():
    # Final Displacement Error
    # FDE_batch = []
    # for i, (x, y, msl_obs, msl_gt) in enumerate(test_dataloader):
    #     x, y, msl_obs, msl_gt = x.to(device), y.to(device), msl_obs.to(device), msl_gt.to(device)
    #     # init = test_data.denorm([x.cpu()])[0][0, -1] # x: [1, 8, 2]
    #     # init = init.numpy()
    #     # tracker = Tracker(init_lat=init[0], init_lon=init[1], init_time=batch.metadata.time[0])
    #
    #     try:
    #         y_p, msl_p = model(x, msl_obs)
    #     except NoEyeException:
    #         logger.info(f"FDE of batch {i+1}/{len(test_dataloader)}: {NoEyeException}")
    #         continue
    #
    #     y, y_p = test_data.denorm_traj([y.cpu(), y_p.cpu()])
    #     # y_p = tracker.results()
    #     # y_p = np.array([y_p.lat[1:], y_p.lon[1:]])[None] # [1, 2, 12]
    #     # y_p = np.transpose(y_p, (0, 2, 1))  # [1, 12, 2]
    #
    #     for j in range(y.size(0)):
    #         FDE = torch.zeros(horizon)
    #         for t in range(horizon):
    #             FDE[t] = haversine(y_p[j, t, 0], y_p[j, t, 1], y[j, t, 0], y[j, t, 1])
    #         FDE_batch.append(FDE)
    #
    #     logger.info(f"FDE of batch {i+1}/{len(test_dataloader)}: {torch.mean(sum(FDE_batch[-y.size(0): ]) / y.size(0))}")
    #
    # FDE_batch = torch.stack(FDE_batch, dim=0)
    # FDE_batch = torch.mean(FDE_batch, dim=0)
    #
    # logger.info(f"Test FDE:")
    # for i in range(horizon):
    #     logger.info(f"{(i + 1) * 6}h: {FDE_batch[i]:.2f}")


    # Visualization
    for i in [0, 20, 43, 60, 80, 100]:
        x, y, msl_obs, msl_gt = test_data[i]
        x, y, msl_obs, msl_gt = x.to(device), y.to(device), msl_obs.to(device), msl_gt.to(device)
        # init = test_data.denorm([x.cpu()])[0][-1] # x: [8, 2]
        # init = init.numpy()
        # tracker = Tracker(init_lat=init[0], init_lon=init[1], init_time=batch.metadata.time[0])

        # for pred in rollout(model, batch, steps=horizon): # [B, T, 2]
        #     pred = pred.to("cpu")
        #     tracker.step(pred)
        # y_p = tracker.results()
        # y_p = np.array([y_p.lat[1:], y_p.lon[1:]])  # [2, 12]
        # y_p = y_p.transpose()  # [12, 2]
        y_p, msl_p = model(x.unsqueeze(0), msl_obs.unsqueeze(0))
        y_p, msl_p = y_p.squeeze(), msl_p.squeeze()
        print(f"y_p: {y_p.shape}")
        print(y_p)

        x, y, y_p = test_data.denorm_traj([x.cpu(), y.cpu(), y_p.cpu()])

        plot_and_save(x, y, y_p, i, fig_dir)

