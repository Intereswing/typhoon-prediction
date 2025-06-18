import math
import torch

def haversine(lat1, lon1, lat2, lon2):
    # 将十进制度数转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # 地球半径（单位：公里）
    r = 6371.0
    return c * r  # 返回距离，单位：公里


def haversine_torch(x1, x2):
    """
    x1, x2: [B, T, 2]，最后一个维度是 (lat, lon)，单位为“度”
    返回: [B, T]，单位为“米”
    """
    # 拆分为纬度和经度
    lat1 = torch.deg2rad(x1[..., 0])
    lon1 = torch.deg2rad(x1[..., 1])
    lat2 = torch.deg2rad(x2[..., 0])
    lon2 = torch.deg2rad(x2[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.asin(torch.sqrt(a))

    R = 6371.0  # 地球平均半径，单位：千米
    distance = R * c  # [B, T]

    return distance