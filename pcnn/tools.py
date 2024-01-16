import cv2
import numpy as np
def find_brain(img):
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    bg = np.zeros_like(img)
    cv2.drawContours(bg, [largest_contour], 0, 1, -1)
    bg = cv2.erode(bg, np.ones((30,30)), iterations=1)
    return bg

def find_largest_connected_component(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    bg = np.zeros_like(img)
    cv2.drawContours(bg, [largest_contour], 0, 1, -1)
    return bg