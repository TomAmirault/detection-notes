import os, sys
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.image_similarity.resize_minkowski_interpolation import minkowski_resize
from datetime import datetime
from logger_config import setup_logger, save_fig_with_limit
from pathlib import Path
import logging
from logger_config import setup_logger


# HYPERPARAMETERS

# Selectivity of matches between keypoints
topMatchesFactor = 0.3
# Threshold to binarize grayscale images
gray_threshold = 140
# Order of the Minkowski mean used for pixel interpolation in image resizing
minkowski_mean_order = 4
# Threshold on pixel differences to consider them significant (depend on minkowski_mean_order)
diff_threshold = 115
# Shape to which images are resized before difference calculation
shape_of_diff = (40, 40)        


def save_comparison(matches_drawn : np.ndarray, img2_aligned : np.ndarray, overlay : np.ndarray, gray1 : np.ndarray, gray2 : np.ndarray, gray1_mink : np.ndarray, gray2_mink : np.ndarray, diff : np.ndarray, isSimilar : bool, transfo : bool):
    """ Saves the history of the transformations applied to the images to compare them in logs/image-comparison/.
    Can be used to finetune the hyperparameters.

    Args:
        matches_drawn (np.ndarray): image of the matches between keypoints of both images
        img2_aligned (np.ndarray): aligned new image after homography transformation if transfo is True, original new image otherwise
        overlay (np.ndarray): overlay of both images
        gray1 (np.ndarray): thresholded reference image
        gray2 (np.ndarray): thresholded new image
        gray1_mink (np.ndarray): preprocessed reference image
        gray2_mink (np.ndarray): preprocessed new image
        diff (np.ndarray): difference image
        isSimilar (bool): indicates if the images are considered similar
        transfo (bool): indicates if a transformation was applied
    """
    fig, axes = plt.subplots(2, 4, figsize=(8, 8))

    axes[0, 0].imshow(matches_drawn)
    axes[0, 0].set_title('Matches')
    axes[0, 1].imshow(img2_aligned)
    axes[0, 1].set_title('New image')
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Overlay of both')

    im1 = axes[0, 3].imshow(gray1, cmap='gray')
    axes[0, 3].set_title('Thresholded old image')
    im2 = axes[1, 0].imshow(gray2, cmap='gray')
    axes[1, 0].set_title('Thres. new image')
    im3 = axes[1, 1].imshow(gray1_mink, cmap='gray')
    axes[1, 1].set_title('Preprocessed old image')
    im4 = axes[1, 2].imshow(gray2_mink, cmap='gray')
    axes[1, 2].set_title('Prepr. new image')

    imd = axes[1, 3].imshow(diff, cmap='gray')
    axes[1, 3].set_title('Difference')

    fig.colorbar(im1, ax=axes[0, 3], orientation='vertical')
    fig.colorbar(im2, ax=axes[1, 0], orientation='vertical')
    fig.colorbar(im3, ax=axes[1, 1], orientation='vertical')
    fig.colorbar(im4, ax=axes[1, 2], orientation='vertical')
    fig.colorbar(imd, ax=axes[1, 3], orientation='vertical')

    if transfo:
        if isSimilar:
            fig.suptitle("[Visual Comparison] Both images are similar after alignment")
        else:
            fig.suptitle("[Visual Comparison] After alignment, both images are still different")
    else:
        if isSimilar:
            fig.suptitle("[Visual Comparison] Both images are similar without any transformation")
        else:
            fig.suptitle("[Visual Comparison] Without transformation, both images are different")

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{datetime.now().microsecond//1000:03d}"
    file_name =f"logs/image-comparison/comparison_{stamp}.jpg"
    save_fig_with_limit(file_name, fig)
    plt.close(fig)



def isSimilar(old_img_path:Path, new_img_path:Path) -> bool:
    """This function compares two images to determine if they're visually similar by first checking direct pixel
    differences, then if needed, using ORB feature detection to align the images via homography transformation
    before performing a second similarity comparison.

    Args:
        old_img_path (Path): path to the reference image
        new_img_path (Path): path to the new image

    Returns:
        bool: indicating if the images are of the same sheet of paper
    """    
    logger = setup_logger("isSimilar")
    try:
        old_img = cv2.imread(old_img_path)
        new_img = cv2.imread(new_img_path)
        img1 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

         # ---- DIFFERENCES WITHOUT (W/O) TRANSFORMATIONS ----

        overlay_wo = np.zeros_like(old_img)
        height, width, channels = old_img.shape
        new_img_overlay = cv2.resize(new_img, (width, height))
        cv2.addWeighted(old_img, 0.5, new_img_overlay, 0.5, 0, overlay_wo)

        _, gray1_wo = cv2.threshold(img1, gray_threshold, 255, cv2.THRESH_BINARY)
        _, gray2_wo = cv2.threshold(img2, gray_threshold, 255, cv2.THRESH_BINARY)

        # We invert the colors to give more importance to the black pixels (writings) when applying the Minkowski mean
        gray1_wo_inv = 255 - gray1_wo
        gray2_wo_inv = 255 - gray2_wo
        h = min(gray1_wo_inv.shape[0], gray2_wo_inv.shape[0])
        w = min(gray1_wo_inv.shape[1], gray2_wo_inv.shape[1])
        # We resize the images to an identical shape using interpolation based on Minkowski mean
        gray1_mink_wo = minkowski_resize(gray1_wo_inv, shape_of_diff, minkowski_mean_order)
        gray2_mink_wo = minkowski_resize(gray2_wo_inv, shape_of_diff, minkowski_mean_order)

        diff_wo = cv2.absdiff(gray1_mink_wo, gray2_mink_wo)
        diff_cut_wo = (diff_wo>diff_threshold) * diff_wo
        answer_wo = np.all(diff_cut_wo<=diff_threshold)
        logger.debug(os.path.basename(old_img_path) + " and " + os.path.basename(new_img_path) + " identical before transformations ? " + str(answer_wo))

        if logger.level == logging.DEBUG:
            save_comparison(old_img, new_img, overlay_wo, gray1_wo, gray2_wo, gray1_mink_wo, gray2_mink_wo, diff_cut_wo, answer_wo, transfo=False)
        if answer_wo:
            return True

        # ---- ALIGNMENT ----

        # - Keypoint detection
        orb = cv2.ORB_create(nfeatures = 1000)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        if descriptors1 is None or descriptors2 is None:
            logger.debug("Not enough keypoints found to compare " + os.path.basename(old_img_path) + " and " + os.path.basename(new_img_path))
            return False

        # - Keypoint matching
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        all_matches = list(matcher.match(descriptors1, descriptors2, None))
        sorted_matches = sorted(all_matches, key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(sorted_matches) * topMatchesFactor)
        matches = sorted_matches[:numGoodMatches]

        matches_printed = cv2.drawMatches(old_img,keypoints1,new_img,keypoints2,matches,None)

        # - Transformation computation
        points1 = np.zeros((len(matches), 2), dtype = np.float32)
        points2 = np.zeros((len(matches), 2), dtype = np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        height, width, channels = old_img.shape
        img2_transfo = cv2.warpPerspective(new_img, h, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        overlay = np.zeros_like(old_img)
        cv2.addWeighted(old_img, 0.5, img2_transfo, 0.5, 0, overlay)


        # ---- DIFFERENCES AFTER TRANSFORMATION ----

        gray1 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_transfo, cv2.COLOR_BGR2GRAY)
        _, gray1 = cv2.threshold(gray1, gray_threshold, 255, cv2.THRESH_BINARY)
        _, gray2 = cv2.threshold(gray2, gray_threshold, 255, cv2.THRESH_BINARY)

        gray1_inv = 255 - gray1
        gray2_inv = 255 - gray2
        h = min(gray1_inv.shape[0], gray2_inv.shape[0])
        w = min(gray1_inv.shape[1], gray2_inv.shape[1])
        gray1_mink = minkowski_resize(gray1_inv, shape_of_diff, minkowski_mean_order)
        gray2_mink = minkowski_resize(gray2_inv, shape_of_diff, minkowski_mean_order)

        diff = cv2.absdiff(gray1_mink, gray2_mink)
        diff_cut = (diff>diff_threshold) * diff
        answer = np.all(diff<=diff_threshold)
        logger.debug(os.path.basename(old_img_path) + " and " + os.path.basename(new_img_path) + " identical after transformations ? " + str(answer))

        if logger.level == logging.DEBUG:
            save_comparison(matches_printed, img2_transfo, overlay, gray1, gray2, gray1_mink, gray2_mink, diff_cut, answer, transfo=True)
        return answer.item()
    except Exception as e:
        logger.error("Exception during the execution of isSimilar: " + str(e))
        return False

if __name__ == '__main__':
    tmp_dir = os.path.join(REPO_PATH, "tmp/test-similarity")
    files = os.listdir(tmp_dir)
    files = [f for f in files if f.endswith('.jpg')]
    
    print("Longueur de files :", len(files))

    im0_path = os.path.join(tmp_dir, "detection_20251022-102208-532_q0.jpg")

    for f in files:
        print("L'image traitée est :", f)
        filename = os.path.join(tmp_dir, f)
        res = isSimilar(im0_path, filename)
        print("Resultat de la comparaison :", res)