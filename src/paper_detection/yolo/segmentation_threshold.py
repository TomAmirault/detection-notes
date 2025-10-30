'''This module provides functions to detect a sheet of paper on an image, segment it, and correct its perspective.'''
import numpy as np
import cv2
import os
import sys
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
from src.paper_detection.edges_based.perspective_corrector import order_corners, get_output_size


def corrected_perspective_white(img:np.ndarray, corners:np.ndarray) -> np.ndarray:
    """Applies a perspective correction to straighten the sheet, adding a white background where needed.

    Args:
        img (np.ndarray): input image
        corners (np.ndarray): corner points of the detected sheet on the input image

    Returns:
        np.ndarray: corrected perspective image with a white background (if needed)
    """
    corners = order_corners(corners)
    output_width, output_height = get_output_size(corners)
    dst_points = np.array([[0, 0], [output_width - 1, 0],
                          [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)

    transformation_matrix = cv2.getPerspectiveTransform(corners, dst_points)

    corrected = cv2.warpPerspective(
        img, transformation_matrix, (output_width, output_height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return corrected


def get_extreme_points(mask:np.ndarray) -> np.ndarray:
    """Get the nearest points of the binary mask from the 4 corners of the image.

    Args:
        mask (np.ndarray): Binary mask

    Returns:
        np.ndarray: Unordered extreme points of the mask
    """    
    h, w = mask.shape[:2]
    corners_ref = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    extreme_points = []
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)) 
    for corner in corners_ref:
        distances = np.linalg.norm(points - corner, axis=1)
        closest_idx = np.argmin(distances)
        extreme_points.append(points[closest_idx])
    extreme_points = np.array(extreme_points)
    return extreme_points


def get_mask(img : np.ndarray) -> np.ndarray :
    """It applies threshlolding and morphological operations to obtain a binary mask
    of the sheet of paper in the image. The returned mask is the largest connected component
    of the thresholded image.

    Args:
        img (np.ndarray): Frame shot by the camera

    Returns:
        np.ndarray: Binary mask of the sheet of paper
    """     
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 100, 255, type = cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morph_close followed by morph_open is the sequence that loses the less pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
    if num_labels>1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else :
        largest_label = 0
    largest_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def crop_image_around_object(img:np.ndarray, rect:tuple[int]) -> np.ndarray:
    """It is the main function of this module and segments, crops and corrects the perspective of the image.

    Args:
        img (np.ndarray): frame capted by the camera
        rect (tuple): tuple representing the bounding box of the object in the format (x_left, y_top, width, height)

    Returns:
        np.ndarray: the image cropped around the sheet
    """    
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]
    mask = get_mask(roi)
 
    points = np.column_stack(np.where(mask > 0))
    rect = cv2.minAreaRect(points[:, ::-1])
    box = np.int32(cv2.boxPoints(rect))

    # Rotating the mask such it puts the minimum area rectangle bounding the mask vertical, and cropping around it.
    corrected_mask = corrected_perspective_white(mask, box)
    # Computing the four extreme points of the corrected_mask
    extreme_points = get_extreme_points(corrected_mask)
    # Rotating the original image such it puts the minimum area rectangle bounding the mask vertical, and cropping around it.
    corrected_image = corrected_perspective_white(roi, box)
    # Cropping the original image around the extreme points and correcting the perspective such it is a rectangle.
    cropped = corrected_perspective_white(corrected_image, extreme_points)

    return cropped


def get_binary_image_of_text(img:np.ndarray, rect:tuple) -> np.ndarray:
    """This function goes further the man function crop_image_around_object by thresholding the writings
    on the sheet of paper.

    Args:
        img (np.ndarray): Frame shot by the camera
        rect (tuple): Tuple representing the bounding box of the object in the format (x_left, y_top, width, height)

    Returns:
        np.ndarray: Binary image of the text on the sheet of paper
    """  
    corrected_masked_image = crop_image_around_object(img, rect)
    gray = cv2.cvtColor(corrected_masked_image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return thresh_image