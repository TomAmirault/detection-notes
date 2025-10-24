# Configuration of parameters used for leaf detection via contour detection
# Returns the parameters to be updated in shape_detector.py after configuration

import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
import cv2
import numpy as np
import time
from src.proc.paper_detection.shape_detector import shape_detector


def tune_shape_detector_camera() -> dict[str, float]:
    """
    Interactive tuner for shape_detector
    """

    # Default values (sufficient for most cases)
    BILATERAL_D = 5
    BILATERAL_SIGMA_COLOR = 20
    BILATERAL_SIGMA_SPACE = 20
    CANNY_THRESHOLD1 = 75
    CANNY_THRESHOLD2 = 200
    MORPH_KERNEL_SIZE = 7
    MORPH_ITERATIONS = 1
    POLY_EPSILON_FACTOR = 0.01
    MIN_AREA_RATIO = 0.05
    MAX_SATURATION = 100
    MIN_VALUE = 175

    def nothing(x):
        pass

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Shape Detector Tuner", cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("D", "Shape Detector Tuner", BILATERAL_D, 20, nothing)
    cv2.createTrackbar("SIG_COLOR", "Shape Detector Tuner", BILATERAL_SIGMA_COLOR, 200, nothing)
    cv2.createTrackbar("SIG_SPACE", "Shape Detector Tuner", BILATERAL_SIGMA_SPACE, 200, nothing)
    cv2.createTrackbar("THRESHOLD1", "Shape Detector Tuner", CANNY_THRESHOLD1, 300, nothing)
    cv2.createTrackbar("THRESHOLD2", "Shape Detector Tuner", CANNY_THRESHOLD2, 300, nothing)
    cv2.createTrackbar("KERNEL_SIZE", "Shape Detector Tuner", MORPH_KERNEL_SIZE, 21, nothing) 
    cv2.createTrackbar("ITERATIONS", "Shape Detector Tuner", MORPH_ITERATIONS, 10, nothing)
    cv2.createTrackbar("EPSILON", "Shape Detector Tuner", int(POLY_EPSILON_FACTOR*100), 20, nothing)
    cv2.createTrackbar("MIN_AREA", "Shape Detector Tuner", int(MIN_AREA_RATIO*100), 50, nothing)
    cv2.createTrackbar("MAX_SAT", "Shape Detector Tuner", MAX_SATURATION, 255, nothing)
    cv2.createTrackbar("MIN_VALUE", "Shape Detector Tuner", MIN_VALUE, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Safe reading of the trackbar values
        bilateral_d = max(cv2.getTrackbarPos("D", "Shape Detector Tuner"), 1)
        bilateral_sigma_color = cv2.getTrackbarPos("SIG_COLOR", "Shape Detector Tuner")
        bilateral_sigma_space = cv2.getTrackbarPos("SIG_SPACE", "Shape Detector Tuner")
        canny_threshold1 = cv2.getTrackbarPos("THRESHOLD1", "Shape Detector Tuner")
        canny_threshold2 = cv2.getTrackbarPos("THRESHOLD2", "Shape Detector Tuner")
        morph_kernel_size = max(cv2.getTrackbarPos("KERNEL_SIZE", "Shape Detector Tuner"), 1)
        if morph_kernel_size % 2 == 0:
            morph_kernel_size += 1
        morph_iterations = max(cv2.getTrackbarPos("ITERATIONS", "Shape Detector Tuner"), 1)
        poly_epsilon_factor = max(cv2.getTrackbarPos("EPSILON", "Shape Detector Tuner") / 100, 0.001)
        min_area_ratio = max(cv2.getTrackbarPos("MIN_AREA", "Shape Detector Tuner") / 100, 0.001)
        max_saturation = cv2.getTrackbarPos("MAX_SAT", "Shape Detector Tuner") 
        min_value = cv2.getTrackbarPos("MIN_VALUE", "Shape Detector Tuner")

        # Intermediate processing steps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preproc = cv2.bilateralFilter(gray, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
        edges = cv2.Canny(preproc, canny_threshold1, canny_threshold2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        shapes = shape_detector(frame, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                                canny_threshold1, canny_threshold2, morph_kernel_size, morph_iterations,
                                poly_epsilon_factor, min_area_ratio, max_saturation, min_value)

        # Display contours on a copy of the image
        detected = frame.copy()
        for s in shapes:
            cv2.drawContours(detected, [s], -1, (0, 255, 0), 4)

        # Resize all steps for a 2x2 canvas
        h, w = frame.shape[:2]
        preproc_color = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        closed_color = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

        preproc_resized = cv2.resize(preproc_color, (w//2, h//2))
        edges_resized = cv2.resize(edges_color, (w//2, h//2))
        closed_resized = cv2.resize(closed_color, (w//2, h//2))
        detected_resized = cv2.resize(detected, (w//2, h//2))

        # Combine into a 2x2 canvas
        top = np.hstack((preproc_resized, edges_resized))
        bottom = np.hstack((closed_resized, detected_resized))
        canvas = np.vstack((top, bottom))
 
        cv2.imshow("Shape Detector Tuner", canvas)

        if cv2.waitKey(30) == ord('q'):
            break
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

    # Return the final configuration
    final_config = {
        "BILATERAL_D": bilateral_d,
        "BILATERAL_SIGMA_COLOR": bilateral_sigma_color,
        "BILATERAL_SIGMA_SPACE": bilateral_sigma_space,
        "CANNY_THRESHOLD1": canny_threshold1,
        "CANNY_THRESHOLD2": canny_threshold2,
        "MORPH_KERNEL_SIZE": morph_kernel_size,
        "MORPH_ITERATIONS": morph_iterations,
        "POLY_EPSILON_FACTOR": poly_epsilon_factor,
        "MIN_AREA_RATIO": min_area_ratio,
        "MAX_SATURATION": max_saturation,
        "MIN_VALUE": min_value,
    }
    return final_config


if __name__ == "__main__":
    cfg = tune_shape_detector_camera()
    print("Configuration finale :", cfg)
