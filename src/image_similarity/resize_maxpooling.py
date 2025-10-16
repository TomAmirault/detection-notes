import numpy as np
import cv2

def resize_maxpool(img, new_shape, p=2):
    """
    img: image 2D (grayscale)
    new_shape: (new_height, new_width)
    """
    H, W = img.shape
    new_H, new_W = new_shape

    # Initialiser la nouvelle image
    new_img = np.zeros((new_H, new_W), dtype=img.dtype)

    # Calcul de la taille de chaque bloc
    h_step = H // new_H
    w_step = W // new_W
    
    for i in range(new_H):
        for j in range(new_W):
            block = img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step].astype(np.float32)
            # print(block.shape)
            new_img[i, j] = new_img[i, j] = (np.mean(block**p))**(1/p)
            # print("max du block :", new_img[i, j])
    return new_img.astype(img.dtype)

# img = cv2.imread("/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/src/recog/test_set/IMG_1702.jpeg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resized = resize_maxpool(gray, (80,80),  2)
# print(np.min(resized))
# print(np.max(resized))

block = np.array([[10, 100, 200]], dtype=np.float32)
for p in [1, 2, 4, 10]:
    val = (np.mean(block ** p)) ** (1/p)
    print(f"p={p}, Minkowski={val}")