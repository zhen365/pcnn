import numpy as np
import cv2
from scipy.signal import find_peaks_cwt

from .tools import find_largest_connected_component

def calculate_otsu_threshold_in_mask1(image_gray, mask_gray):
    masked_image = image_gray * mask_gray
    inter_class_variance = 0
    best_threshold = 0

    for threshold in range(256):
        _, thresh = cv2.threshold(masked_image, threshold, 255, cv2.THRESH_BINARY)
        thresh = (thresh/255).astype(np.uint8)
        if np.sum(thresh) == 0:
            continue
        fore_mask = thresh
        back_mask = mask_gray - fore_mask
        # 计算背景和前景的像素概率
        foreground_prob = np.sum(fore_mask) / np.sum(mask_gray)
        background_prob = np.sum(back_mask) / np.sum(mask_gray)
        # 计算背景和前景的平均灰度值
        background_mean = np.sum(fore_mask*masked_image) / (np.sum(fore_mask) + 1e-6)
        foreground_mean = np.sum(back_mask*masked_image) / (np.sum(back_mask) + 1e-6)
        # 计算类间方差
        current_variance = background_prob * foreground_prob * (background_mean - foreground_mean) ** 2
        # 更新最佳阈值和类间方差
        if current_variance > inter_class_variance:
            inter_class_variance = current_variance
            best_threshold = threshold

    return best_threshold

def calculate_otsu_threshold_in_mask2(image_gray, mask_gray):
    masked_image = image_gray * mask_gray
    inter_class_variance = 0
    best_threshold = 0

    for threshold in range(256):
        _, thresh = cv2.threshold(masked_image, threshold, 255, cv2.THRESH_BINARY)
        if np.sum(thresh) == 0:
            continue
        fore_mask = find_largest_connected_component(thresh)
        back_mask = mask_gray - fore_mask
        # 计算背景和前景的像素概率
        foreground_prob = np.sum(fore_mask) / np.sum(mask_gray)
        background_prob = np.sum(back_mask) / np.sum(mask_gray)
        # 计算背景和前景的平均灰度值
        background_mean = np.sum(fore_mask*masked_image) / (np.sum(fore_mask) + 1e-6)
        foreground_mean = np.sum(back_mask*masked_image) / (np.sum(back_mask) + 1e-6)
        # 计算类间方差
        current_variance = background_prob * foreground_prob * (background_mean - foreground_mean) ** 2
        # 更新最佳阈值和类间方差
        if current_variance > inter_class_variance:
            inter_class_variance = current_variance
            best_threshold = threshold

    return best_threshold

def calculate_otsu_threshold_in_mask3(image_gray, mask_gray):
    hist = cv2.calcHist([image_gray*mask_gray], [0], mask_gray, [256], [0, 256])
    index = np.where(hist.ravel() != 0)
    width = index[0][-1] - index[0][0] +1
    peaks_index_p = find_peaks_cwt(hist.ravel(), np.arange(1, width))
    peaks_index_n = find_peaks_cwt(-hist.ravel(), np.arange(1, width))
    threshold = int((peaks_index_n[0]+peaks_index_p[0])/2)
    return threshold