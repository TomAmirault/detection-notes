"""It provides utilities to evaluate the blurriness of individual images as well as to find the least blurry frame in a video."""

import cv2
import numpy as np
from ultralytics.engine.results import Results
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def laplacian_variance(img:np.ndarray, grid_size: tuple[int]=(40, 40), top_k: int=100) -> float:
    """Fragments the image into a grid and computes the average of the top_k lowest
    variances of the Laplacian for each grid cell to assess blurriness.

    Args:
        img (np.ndarray): the frame we evaluate the blurriness of
        grid_size (tuple, optional): the number of rows and columns to divide the image into. Defaults to (40, 40).
        top_k (int, optional): the number of lowest variance scores to average. Defaults to 100.

    Returns:
        float: the value of the estimated bluriness (lower means more blurry)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    n_rows, n_cols = grid_size
    dh, dw = h // n_rows, w // n_cols

    scores = []
    for i in range(n_rows):
        for j in range(n_cols):
            y1, y2 = i * dh, (i + 1) * dh
            x1, x2 = j * dw, (j + 1) * dw
            roi = gray[y1:y2, x1:x2]

            score = cv2.Laplacian(roi, cv2.CV_64F, ksize = 3).var()
            scores.append(score)
    sorted_scores = sorted(scores)
    top_scores = sorted_scores[:min(top_k, len(sorted_scores))]
    avg_blurriness = np.mean(top_scores)

    return avg_blurriness

def less_blurred(video:list[Results]) -> int:
    """Finds the index of the least blurry image in the video using laplacian_variance.

    Args:
        video (list[Results]): the list of the output of the YOLO model on the frames captured by the camera

    Returns:
        int: the index of the least blurry frame
    """
    best = 0
    best_value = laplacian_variance(video[0].orig_img)
    for i in range(1,len(video)):
        value_tmp = laplacian_variance(video[i].orig_img)
        if value_tmp>best_value:
            best_value=value_tmp
            best = i
    return best

def less_blurred_roi(video:list[Results]) -> int:
    """Finds the index of the image in the video where the bounding box is the least blurry using laplacian_variance.

    Args:
        video (list[Results]): the list of the output of the YOLO model on the frames captured by the camera

    Returns:
        int: the index of the least blurry frame
    """
    best = 0
    x_center, y_center, w, h = video[0].boxes.xywh[0]
    best_value = laplacian_variance(video[0].orig_img[int(y_center-h/2):int(y_center+h/2), int(x_center-w/2):int(x_center+w/2)])
    for i in range(1,len(video)):
        x_center, y_center, w, h = video[i].boxes.xywh[0]
        value_tmp = laplacian_variance(video[i].orig_img[int(y_center-h/2):int(y_center+h/2), int(x_center-w/2):int(x_center+w/2)])
        if value_tmp>best_value:
            best_value=value_tmp
            best = i
    return best