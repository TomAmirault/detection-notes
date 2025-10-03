import numpy as np
import cv2
import matplotlib.pyplot as plt
import ot

# source0 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1689.jpeg"), 1689]

source1 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1690.jpeg"), 1690]
source2 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1691.jpeg"), 1691]
source3 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1692.jpeg"), 1692]
source4 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1693.jpeg"), 1693]

sample0 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1694.png"), 1694]
sample1 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1695.png"), 1695]
sample2 = [cv2.imread("/Users/noahparisse/Downloads/IMG_1696.png"), 1696]
# Variables de l'ORB
topMatchesFactor = 0.1       # Sélectivité des matches de descriptors

# Charger et prétraiter une image avec OpenCV
def preprocess_image(img, size=(64, 64)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # charger en niveaux de gris
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # redimensionner
    img = img.astype(np.float64)
    img = img / img.sum()  # normalisation en probabilité
    return img

# Calculer la distance de Wasserstein entre deux images
def wasserstein_distance(img1, img2):
    a = img1.flatten()
    b = img2.flatten()

    # Grille de coordonnées (pixels)
    n = img1.shape[0]
    coords = np.array([(i, j) for i in range(n) for j in range(n)])
    M = ot.dist(coords, coords, metric='euclidean')  # matrice des coûts
    M /= M.max()  # normalisation

    # Transport optimal (Earth Mover's Distance)
    distance = ot.emd2(a, b, M)
    return distance

def compare(sourcea, sourceb):
    img1 = preprocess_image(sourcea[0])
    img2 = preprocess_image(sourceb[0])

    d = wasserstein_distance(img1, img2)
    print(f"--------- Distance de Wasserstein entre les images {sourcea[1]} et {sourceb[1]}: {d:.4f}")

    # Visualisation
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Image 1")
    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Image 2")
    plt.show()

def realign(sourcea, sourceb):
    img1 = cv2.cvtColor(sourcea[0], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(sourceb[0], cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures = 1000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Image1", img1_with_keypoints)
    # img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Image 2", img2_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(descriptors1, descriptors2, None)
    matches = list(matches)
    sorted_matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(sorted_matches) * topMatchesFactor)
    matches = sorted_matches[:numGoodMatches]

    # im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    # cv2.imshow("Image matches", im_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, channels = sourcea[0].shape
    img2_reg = cv2.warpPerspective(sourceb[0], h, (width, height))
    cv2.imshow("Image1", sourcea[0])
    cv2.imshow("Image2", sourceb[0])  
    # cv2.imshow("Image2 realigned", img2_reg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    color1 = sourcea[0].copy()
    color2 = img2_reg

    overlay = sourcea[0].copy()
    cv2.addWeighted(sourcea[0], 0.5, img2_reg, 0.5, 0, overlay)
    cv2.imshow("Images overlaid", overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    h = min(color1.shape[0], color2.shape[0])
    w = min(color1.shape[1], color2.shape[1])
    color1 = cv2.resize(color1, (w, h))
    color2 = cv2.resize(color2, (w, h))

    diff = cv2.absdiff(color1, color2)
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# compare(source1, source2)
# compare(source1, source3)
# compare(source2, source3)
# compare(source1, source4)
# compare(source2, source4)
# compare(source3, source4)

# compare(source0, source1)

realign(source3, source2)