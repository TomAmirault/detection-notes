import cv2
import numpy as np

def warpPerspective(image_path, save_path):

    img = cv2.imread(image_path)

    h, w = img.shape[:2]
    
    edges = cv2.Canny(img, threshold1=100, threshold2=200)

   
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    max_area = 0
    best_box = None

    for cnt in contours:
        if len(cnt) < 5:
            continue
        
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        area = rw * rh
        
        if area > max_area:
            max_area = area
            best_box = cv2.boxPoints(rect)
            best_box = best_box.astype(int)
            
    img_biggest = img.copy()
    if best_box is not None:
        cv2.drawContours(img_biggest, [best_box], 0, (0, 255, 0), 3)


    # Transformation de perspective pour adapter à la taille de l'image
    if best_box is not None:
        # Ordre des points pour éviter les inversions
        best_box = best_box[np.argsort(best_box[:,1])]  # trier par y
        top = sorted(best_box[:2], key=lambda x: x[0])
        bottom = sorted(best_box[2:], key=lambda x: x[0])
        ordered_box = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

        # Destination : rectangle de même taille que l'image
        dst_pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")

        # Calcul de la matrice de transformation
        M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
        warped = cv2.warpPerspective(img, M, (w,h))

        
        cv2.imwrite(save_path, warped)

        return save_path
