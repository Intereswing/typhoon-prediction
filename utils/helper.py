import logging
import pickle
from datetime import datetime

import xarray as xr
import torch
import numpy as np
from aurora import Batch, Metadata


def setup_logger(log_dir, log_name):
    """Set up logger of file and console.

    :param log_dir: Directory containing log file.
    :param log_name: Name of log file.
    :return: logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_dir / f"{log_name}.log")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def preprocess_and_pad_era5(era5_0: xr.Dataset, era5_1: xr.Dataset, var_name: str) -> torch.Tensor:
    """
    Preprocess ERA5 datasets for Aurora Batch. It is able to preprocess both atmospheric and surface variables.

    :param era5_0: The ERA5 dataset at time 0.
    :param era5_1: The ERA5 dataset at time 1. (Next time step)
    :param var_name: Variable name to extract.
    :return: Padded tensor with shape [1, 2, C, 721, 1440]
    """
    var_0 = era5_0[var_name].values[None, None] # [C, 240, 320] -> [1, 1, C, 240, 320]
    var_1 = era5_1[var_name].values[None, None] # [C, 240, 320] -> [1, 1, C, 240, 320]
    var = np.concatenate([var_0, var_1], axis=1) # [1, 2, C, 240, 320]
    if var.ndim == 5:
        pad_width = ((0,0),(0,0),(0,0),(121, 360), (400, 720))
    else:
        pad_width = ((0,0),(0,0),(121, 360), (400, 720))

    var_pad = np.pad(
        var,
        pad_width=pad_width,
        mode="constant",
        constant_values=0
    ) # [1, 2, C, 721, 1440]
    return torch.tensor(var_pad)


def preprocess_era5(era5_0: xr.Dataset, era5_1: xr.Dataset, var_name: str) -> torch.Tensor:
    var_0 = era5_0[var_name].values[None, None] # [C, 721, 1440] -> [1, 1, C, 721, 1440]
    var_1 = era5_1[var_name].values[None, None] # [C, 721, 1440] -> [1, 1, C, 721, 1440]
    var = np.concatenate([var_0, var_1], axis=1) # [1, 2, C, 721, 1440]

    return torch.from_numpy(var)


def preprocess_hres_t0(hres_0: xr.Dataset, hres_1: xr.Dataset, var_name: str) -> torch.Tensor:

    var_0 = hres_0[var_name].values[..., ::-1, :][None, None].copy() # [C, 721, 1440] -> [1, 1, C, 721, 1440]
    var_1 = hres_1[var_name].values[..., ::-1, :][None, None].copy() # [C, 721, 1440] -> [1, 1, C, 721, 1440]
    var = np.concatenate([var_0, var_1], axis=1) # [1, 2, C, 721, 1440]

    return torch.from_numpy(var)


def get_aurora_batch_era5(data_t0, data_t_1, t0, data_dir):
    with open(data_dir / "era5/aurora-0.25-static.pickle", "rb") as f:
        static_vars = pickle.load(f)

    batch = Batch(
        surf_vars={
            "2t": preprocess_era5(data_t_1, data_t0, "2m_temperature"),
            "10u": preprocess_era5(data_t_1, data_t0, "10m_u_component_of_wind"),
            "10v": preprocess_era5(data_t_1, data_t0, "10m_v_component_of_wind"),
            "msl": preprocess_era5(data_t_1, data_t0, "mean_sea_level_pressure"),
        },
        atmos_vars={
            "z": preprocess_era5(data_t_1, data_t0, "geopotential"),
            "u": preprocess_era5(data_t_1, data_t0, "u_component_of_wind"),
            "v": preprocess_era5(data_t_1, data_t0, "v_component_of_wind"),
            "t": preprocess_era5(data_t_1, data_t0, "temperature"),
            "q": preprocess_era5(data_t_1, data_t0, "specific_humidity"),
        },
        static_vars={
            k: torch.tensor(v) for k, v in static_vars.items()
        },
        metadata=Metadata(
            lat=torch.from_numpy(data_t0["latitude"].values),
            lon=torch.from_numpy(data_t0["longitude"].values),
            time=(datetime.strptime(t0, "%Y%m%d%H"),),
            atmos_levels=tuple(int(level) for level in data_t0["level"].values),
        )
    )

    return batch


def get_aurora_batch_hres_t0(data_t0, data_t_1, t0):
    with open("/data/jiyilun/typhoon/download/era5/aurora-0.25-static.pickle", "rb") as f:
        static_vars = pickle.load(f)

    batch = Batch(
        surf_vars={
            "2t": preprocess_hres_t0(data_t_1, data_t0, "2m_temperature"),
            "10u": preprocess_hres_t0(data_t_1, data_t0, "10m_u_component_of_wind"),
            "10v": preprocess_hres_t0(data_t_1, data_t0, "10m_v_component_of_wind"),
            "msl": preprocess_hres_t0(data_t_1, data_t0, "mean_sea_level_pressure"),
        },
        atmos_vars={
            "z": preprocess_hres_t0(data_t_1, data_t0, "geopotential"),
            "u": preprocess_hres_t0(data_t_1, data_t0, "u_component_of_wind"),
            "v": preprocess_hres_t0(data_t_1, data_t0, "v_component_of_wind"),
            "t": preprocess_hres_t0(data_t_1, data_t0, "temperature"),
            "q": preprocess_hres_t0(data_t_1, data_t0, "specific_humidity"),
        },
        static_vars={
            k: torch.tensor(v) for k, v in static_vars.items()
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, 721),
            lon=torch.linspace(0, 360, 1440 + 1)[:-1],
            time=(datetime.strptime(t0, "%Y%m%d%H"),),
            atmos_levels=tuple(int(level) for level in data_t0["geopotential"]["level"].values),
        )
    )

    return batch


def exclude_pretrained_params(state_dict, pretrained_prefix='pretrained_model'):
    """
    从 state_dict 中排除所有以 pretrained_prefix 开头的参数。
    """
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith(pretrained_prefix + '.')
    }
    return filtered_state_dict
