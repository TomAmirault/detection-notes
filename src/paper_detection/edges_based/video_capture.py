"""
Main script for paper detection using edge-based shape detection
"""

import sys, os
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
import cv2
from src.paper_detection.edges_based.shape_detector import shape_detector
from src.paper_detection.edges_based.save_detection import save_detection


# CAMERA SOURCE (0 for default webcam)
CAMERA_SOURCE = 0
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera: {CAMERA_SOURCE}")

while True:
    ret, img = cap.read()

    # Verify that a frame was captured
    if not ret or img is None:
        continue  
    
    possible_papers = shape_detector(img)

    # Copy image for display
    img_show = img.copy()
    cv2.drawContours(img_show, possible_papers, -1, (0, 255, 0), 2)
    cv2.putText(img_show,
        f"Detected: {len(possible_papers)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if len(possible_papers) > 0 else (0, 0, 255),
        2
    )

    # Save detected sheets with perspective correction
    if len(possible_papers) > 0:
        save_detection(img, possible_papers)

    cv2.imshow("Video Feed", img_show)

    # Press "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
