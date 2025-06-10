from matplotlib import pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_and_save(x, y, y_p, idx, savepath):
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
    ax.scatter(x[:, 1], x[:, 0], color='green', s=3, label="Observed")
    ax.scatter(y_p[:, 1], y_p[:, 0], color='red', s=3, label="Prediction")
    ax.scatter(y[:, 1], y[:, 0], color='blue', s=3, label="Ground truth")

    plt.title(f"Typhoon {idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath / f"typhoon {idx}.png", dpi=300)
    plt.clf()
