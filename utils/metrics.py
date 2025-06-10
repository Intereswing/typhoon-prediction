import math

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
