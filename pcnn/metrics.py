import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

def compute_metrics(mask1, mask2):
    # 计算每个mask的元素数量
    num_mask1 = np.sum(mask1)
    num_mask2 = np.sum(mask2)

    # 计算两个mask的交集
    intersection = np.logical_and(mask1, mask2)
    num_intersection = np.sum(intersection)

    # 计算并集
    union = np.logical_or(mask1, mask2)
    num_union = np.sum(union)

    # 计算Dice系数、Jaccard相似系数
    if num_intersection == 0:
        dice = 0
        jaccard = 0
    else:
        dice = (2 * num_intersection) / (num_mask1 + num_mask2)
        jaccard = num_intersection / num_union

    # 计算ASD
    dist_transform1 = distance_transform_edt(mask1)
    dist_transform2 = distance_transform_edt(mask2)
    surface_distances = np.abs(dist_transform1 - dist_transform2)
    asd = np.mean(surface_distances)

    # 计算HD95
    hd95 = directed_hausdorff(mask1, mask2)[0]

    return ("Dice: %.4f, Jaccard: %.4f, HD95: %.4f, ASD: %.4f" % (dice, jaccard, hd95, asd))