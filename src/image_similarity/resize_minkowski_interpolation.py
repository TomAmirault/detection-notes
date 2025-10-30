import numpy as np

def minkowski_resize(img: np.ndarray, new_shape: tuple[int, int], p: int = 2) -> np.ndarray:
    """Resize an image using an interpolation based on Minkowski mean.

    Args:
        img (np.ndarray): The input image to resize.
        new_shape (tuple[int, int]): The desired output shape (height, width).
        p (int, optional): The order of the Minkowski mean. Defaults to 2.

    Returns:
        np.ndarray: The resized image.
    """    
    h, w = img.shape
    h_dest, w_dest = new_shape

    img_dest = np.zeros((h_dest, w_dest), dtype=img.dtype)

    h_dest_step = h // h_dest
    w_dest_step = w // w_dest

    for i in range(h_dest):
        for j in range(w_dest):
            block = img[i*h_dest_step:(i+1)*h_dest_step, j*w_dest_step:(j+1)*w_dest_step].astype(np.float32)
            img_dest[i, j] = (np.mean(block**p))**(1/p)
    return img_dest.astype(img.dtype)