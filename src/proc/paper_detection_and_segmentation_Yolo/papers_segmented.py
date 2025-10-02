import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_paper_detections(model, img_path, seuil=0.5):
    
    results = model(img_path, imgsz=640, verbose=False)

    for r in results:

        mask = r.boxes.conf >= seuil
        r.boxes = r.boxes[mask]
        if hasattr(r, "masks") and r.masks is not None:
            r.masks = r.masks[mask]

        if len(r.boxes) > 0:
            im = r.plot()
            plt.imshow(im)
            plt.axis("off")
            plt.show()
 
def plot_segmented_papers(model, img_path, seuil=0.5):
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]

    # Faire une prédiction
    results = model(img_path, imgsz=640, verbose=False)

    for r in results:
        mask = r.boxes.conf >= seuil
        r.boxes = r.boxes[mask]
        if hasattr(r, "masks") and r.masks is not None:
            r.masks = r.masks[mask]

        if r.masks is not None and len(r.masks.data) > 0:
            # Boucler sur chaque masque
            for j, m in enumerate(r.masks.data.cpu().numpy()):
                # Redimensionner le masque à la taille originale
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m_resized = (m_resized > 0.5).astype(np.uint8) * 255  # binaire 0/255

                # Appliquer le masque
                masked = cv2.bitwise_and(orig, orig, mask=m_resized)

                # Recadrer autour du masque
                coords = cv2.findNonZero(m_resized)
                x, y, bw, bh = cv2.boundingRect(coords)
                cropped = masked[y:y+bh, x:x+bw]

                # Afficher
                plt.imshow(cropped)
                plt.axis("off")
                plt.show()

            
            
            