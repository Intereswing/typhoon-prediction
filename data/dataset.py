from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import Dataset
from aurora import Batch, Metadata
import xarray as xr
from datetime import datetime

from utils.helper import get_aurora_batch_era5, get_aurora_batch_hres_t0
'''
era5_Western_North_Pacific数据有部分下载时出现损坏
'''


def aurora_collate_fn(batch):
    xs, ys, batches = zip(*batch)
    if len(xs) == 1:
        xs = xs[0].unsqueeze(0)
    else:
        xs = torch.stack(xs, dim=0)

    if len(ys) == 1:
        ys = ys[0].unsqueeze(0)
    else:
        ys = torch.stack(ys, dim=0)
    batches = Batch(
        surf_vars={
            k: torch.cat([x.surf_vars[k] for x in batches], dim=0) for k in batches[0].surf_vars.keys()
        },
        atmos_vars={
            k: torch.cat([x.atmos_vars[k] for x in batches], dim=0) for k in batches[0].atmos_vars.keys()
        },
        static_vars={
            k: v for k, v in batches[0].static_vars.items()
        },
        metadata=Metadata(
            lat=batches[0].metadata.lat,
            lon=batches[0].metadata.lon,
            time=sum([x.metadata.time for x in batches], ()),
            atmos_levels=batches[0].metadata.atmos_levels
        )
    )
    return xs, ys, batches


class TyphoonTrajectoryDataset(Dataset):
    def __init__(
            self,
            data_dir,
            year_start,
            year_end,
            lookback=8,
            horizon=12,
            with_era5_for_aurora=False,
            with_hres_t0_for_aurora=False,
            with_msl=False,
            with_era5=False,
        ):
        self.data_dir = Path(data_dir)
        self.lookback = lookback
        self.horizon = horizon
        self.with_era5_for_aurora = with_era5_for_aurora
        self.with_hres_t0_for_aurora = with_hres_t0_for_aurora
        self.with_msl = with_msl
        self.with_era5 = with_era5 # Only use msl

        self.data_typhoons = {} # {typhoon Path: {time: (lat, lon)}}

        for year in range(year_start, year_end + 1): # 遍历年份。
            for data_file in sorted((self.data_dir / f"bwp{year}").glob("bwp*")): # 遍历一年里的所有台风。
                data_per_typhoon = {} # 使用字典，避免一个时刻有多个位置记录。
                data_per_typhoon_raw = data_file.read_text().splitlines()
                for line in data_per_typhoon_raw:
                    #e.g. WP, 01, 1950050612,   , BEST,   0,  72N, 1508E,  30
                    data_line = [s.strip() for s in line.split(',')]
                    # 只关心 00 06 12 18 小时的数据。
                    if data_line[2][-2:] in ['00', '06', '12', '18']:
                        data_per_typhoon[data_line[2]] = (float(data_line[6][:-1]) / 10, float(data_line[7][:-1]) / 10)

                # 总长度要大于lookback+horizon
                if len(data_per_typhoon) >= lookback + horizon:
                    self.data_typhoons[data_file] = data_per_typhoon


        if not (self.data_dir / "mean_std.pth").exists():
            print(f"Calculate mean and std of lat and lon from {year_start} to {year_end}.")
            self.get_mean_std()
        self.mean, self.std = torch.load(self.data_dir / "mean_std.pth", weights_only=True)

        if self.with_msl:
            self.msl_mean = 1.009578e05
            self.msl_std = 1.332246e03

        if self.with_era5:
            # variable order: [msl]
            self.era5_mean = torch.tensor([1.009578e05])
            self.era5_std = torch.tensor([1.332246e03])

    def __len__(self):
        # 每一条台风对应 len - (lookback + horizon) + 1 条数据
        return sum([len(data_per_typhoon) - (self.lookback + self.horizon) + 1
                    for data_per_typhoon in self.data_typhoons.values()])


    def __getitem__(self, idx):
        """

        :param idx: index.
        :return: [lookback, 2], [horizon, 2].
        """
        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            if idx >= len(data_per_typhoon) - (self.lookback + self.horizon) + 1:
                idx = idx - (len(data_per_typhoon) - (self.lookback + self.horizon) + 1)
            else:
                trajectory = torch.zeros(self.lookback + self.horizon, 2)
                for i in range(self.lookback + self.horizon):
                    trajectory[i] = torch.tensor(list(data_per_typhoon.values())[idx + i])

                # Normalization
                trajectory = (trajectory - self.mean) / self.std

                if not self.with_era5_for_aurora and not self.with_hres_t0_for_aurora and not self.with_msl and not self.with_era5:
                    return trajectory[:self.lookback], trajectory[self.lookback:]

                elif self.with_era5_for_aurora:
                    ts = list(data_per_typhoon.keys())[idx + self.lookback - 2: idx + self.lookback]
                    era5_t0 = xr.open_dataset(self.data_dir / "era5" / f"{ts[1]}.nc")
                    era5_t_1 = xr.open_dataset(self.data_dir / "era5" / f"{ts[0]}.nc") # t = -1
                    batch = get_aurora_batch_era5(era5_t0, era5_t_1, ts[1], self.data_dir)
                    return trajectory[:self.lookback], trajectory[self.lookback:], batch

                elif self.with_hres_t0_for_aurora:
                    ts = list(data_per_typhoon.keys())[idx + self.lookback - 2: idx + self.lookback]
                    hres_t0 = xr.open_dataset(self.data_dir / "hres_t0" / f"{ts[1]}.nc")
                    hres_t_1 = xr.open_dataset(self.data_dir / "hres_t0" / f"{ts[0]}.nc")
                    batch = get_aurora_batch_hres_t0(hres_t0, hres_t_1, ts[1])
                    return trajectory[:self.lookback], trajectory[self.lookback:], batch

                elif self.with_msl:
                    ts = list(data_per_typhoon.keys())[idx: idx + self.lookback + self.horizon]
                    msl = [np.load(self.data_dir / "msl" / f"{t}.npy") for t in ts]
                    msl = np.stack(msl, axis=0)
                    msl = torch.from_numpy(msl)
                    msl = (msl - self.msl_mean) / self.msl_std
                    return (trajectory[:self.lookback], trajectory[self.lookback:],
                            msl[:self.lookback], msl[self.lookback:])

                elif self.with_era5:
                    ts = list(data_per_typhoon.keys())[idx + self.lookback: idx + self.lookback + self.horizon]
                    variable_names = ["mean_sea_level_pressure"]
                    era5_data = [xr.open_dataset(self.data_dir / "era5" / f"{t}.nc")[variable_names].to_array().values for t in ts]
                    era5_data = np.stack(era5_data, axis=0) # [T, C, H, W]
                    era5_data = torch.from_numpy(era5_data)

                    # normalize
                    era5_data = (era5_data - self.era5_mean[..., None, None]) / self.era5_std[..., None, None]

                    return trajectory[:self.lookback], trajectory[self.lookback:], era5_data

        return None


    def get_mean_std(self):
        all_data = []
        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            for point_data in data_per_typhoon.values():
                point = torch.tensor(point_data)
                all_data.append(point)
        all_data = torch.stack(all_data, dim=0)
        print(all_data.shape)
        mean = torch.mean(all_data, dim=0)
        print(mean.shape)
        std = torch.std(all_data, dim=0)
        print(std.shape)
        torch.save((mean, std), self.data_dir / "mean_std.pth")
        print(f"Saved {str(self.data_dir / 'mean_std.pth')}.")


    def get_idx(self, typhoon_num: int, time: str):
        idx = 0
        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            if int(typhoon_file.stem[3:5]) != typhoon_num:
                idx += len(data_per_typhoon) - (self.lookback + self.horizon) + 1
            else:
                idx += list(data_per_typhoon.keys()).index(time) - self.lookback + 1
                # t0, t1, t2, ..., t7
                break
        return idx


    def get_typhoon_name(self, idx: int):
        typhoon_name = None
        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            if idx >= len(data_per_typhoon) - (self.lookback + self.horizon) + 1:
                idx = idx - (len(data_per_typhoon) - (self.lookback + self.horizon) + 1)
            else:
                typhoon_name = typhoon_file.stem
                break

        return typhoon_name


    def denorm_traj(self, x):
        if isinstance(x, list):
            return [x * self.std + self.mean for x in x]
        elif isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        else:
            raise RuntimeError('denorm_traj only accepts list or torch.Tensor.')


    def denorm_msl(self, msl):
        return msl * self.msl_std + self.msl_mean


    def download_era5(self):
        """Download ERA5 data as extra information for development of neural tracker."""
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        era5_ds = xr.open_zarr(url)

        time2download = []
        (self.data_dir / "era5").mkdir(parents=True, exist_ok=True)

        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            time_per_typhoon_need_era5 = list(data_per_typhoon.keys())[self.lookback:]

            for date_time_str in time_per_typhoon_need_era5:
                time2download.append(date_time_str)

                date_time = pd.to_datetime(date_time_str, format="%Y%m%d%H")
                var_names = [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ]

                t0 = datetime.now()
                file_path = self.data_dir / "era5" / (date_time_str + ".nc")
                if not file_path.exists():
                    ds = era5_ds[var_names].sel(time=date_time).compute()
                    t1 = datetime.now()
                    print(f"Download time: {(t1 - t0).total_seconds()}s")
                    ds.to_netcdf(file_path)
                print(f"{date_time_str} variables downloaded!")
                print(f"")
            print(f"{typhoon_file} era5 downloaded!")

        # Download static variables from aurora huggingface repo.
        time2download = set(time2download)
        print(f"{len(time2download)} files processed.")


    def download_hres_t0(self):
        url = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
        hres_ds = xr.open_zarr(url)
        (self.data_dir / "hres_t0").mkdir(parents=True, exist_ok=True)

        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            time_per_typhoon_need_hres = list(data_per_typhoon.keys())[self.lookback - 2: -self.horizon]
            for date_time_str in time_per_typhoon_need_hres:
                date_time = pd.to_datetime(date_time_str, format="%Y%m%d%H")
                var_names = [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ]
                t0 = datetime.now()
                file_path = self.data_dir / "hres_t0" / (date_time_str + ".nc")
                if not file_path.exists():
                    ds = hres_ds[var_names].sel(time=date_time).compute()
                    t1 = datetime.now()
                    print(f"Download time: {(t1 - t0).total_seconds()}s")
                    ds.to_netcdf(file_path)
                print(f"{date_time_str} variables downloaded!")
                print(f"")
            print(f"{typhoon_file} HRES T0 downloaded!")


    def download_msl(self):
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        era5_ds = xr.open_zarr(url)

        (self.data_dir / "msl").mkdir(parents=True, exist_ok=True)
        for typhoon_file, data_per_typhoon in self.data_typhoons.items():
            for time_str in data_per_typhoon.keys():
                date_time = pd.to_datetime(time_str, format="%Y%m%d%H")
                var_names = "mean_sea_level_pressure"
                file_path = self.data_dir / "msl" / f"{time_str}.npy"
                if not file_path.exists():
                    era5_file_path = self.data_dir / "era5" / f"{time_str}.nc"
                    if era5_file_path.exists():
                        ds_this_hour = xr.open_dataset(era5_file_path)
                        msl_this_hour = ds_this_hour[var_names].values
                        np.save(file_path, msl_this_hour)
                        print(f"Save {time_str} msl from era5 folder.")
                    else:
                        t0 = datetime.now()
                        msl_this_hour = era5_ds[var_names].sel(time=date_time).isel(
                            longitude=slice(400, 720),
                            latitude=slice(121, 361)
                        ).compute()
                        t1 = datetime.now()
                        print(f"Download time: {(t1 - t0).total_seconds()}s")
                        np.save(file_path, msl_this_hour.values)
                print(f"{time_str} msl downloaded!")
                print(f"")
            print(f"{typhoon_file} msl downloaded!")


if __name__ == "__main__":
    ds_dir = "/data1/jiyilun/typhoon"
    # calculate mean and std
    ds = TyphoonTrajectoryDataset(ds_dir, 2022, 2022, lookback=8, horizon=12, with_era5=True)
    print(ds.get_typhoon_name(0))
