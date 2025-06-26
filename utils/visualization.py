from matplotlib import pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_and_save(obs_traj, gt_traj, pred_traj, i, dataset, fig_dir):
    lon_min, lon_max = 100, 180
    lat_min, lat_max = 0, 60

    # Earth
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Trajectory
    ax.scatter(obs_traj[:, 1], obs_traj[:, 0], color='green', s=3, label="Observed")
    ax.scatter(pred_traj[:, 1], pred_traj[:, 0], color='red', s=3, label="Prediction")
    ax.scatter(gt_traj[:, 1], gt_traj[:, 0], color='blue', s=3, label="Ground truth")

    typhoon_name = dataset.get_typhoon_name(i)
    plt.title(typhoon_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"typhoon {i} {typhoon_name}.png", dpi=300)
    plt.clf()
