'''Ce n'était pas concluant : pas vraiment de meilleurs résultats au global'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
from image_similarity.resize_minkowski_interpolation import minkowski_resize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Affichage des graphes lors de l'exécution du script ?
print_graphs = True

# Hyperparamètres
nfeaturesORB1 = 1000
nfeaturesORB2 = 1000
topMatchesFactor1 = 0.2       # Sélectivité des matches entre keypoints
topMatchesFactor2 = 0.2
gray_threshold = 125
minkowski_mean_order = 2        # Ordre de la moyenne de Minkowski utilisée pour l'interpolation des pixels dans le redimensionnement des images
diff_threshold = 20     # 20 si Minkowski_mean_order = 2, 8 si = 4, 4 si = 10
shape_of_diff = (40, 40)

def double_alignment(sourcea, sourceb):
    img1 = cv2.cvtColor(sourcea[0], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(sourceb[0], cv2.COLOR_BGR2GRAY)

    ## --------- 1er alignement --------------

    # Détection des keypoints
    orb = cv2.ORB_create(nfeatures = nfeaturesORB1)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    # img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Image1", img1_with_keypoints)
    # img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Image 2", img2_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Matching des keypoints
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    all_matches = list(matcher.match(descriptors1, descriptors2, None))
    sorted_matches = sorted(all_matches, key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(sorted_matches) * topMatchesFactor1)
    matches = sorted_matches[:numGoodMatches]
    # im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    # cv2.imshow("Image matches", im_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Calcul de la transformation
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, channels = sourcea[0].shape
    img2_reg = cv2.warpPerspective(sourceb[0], h, (width, height))
    # cv2.imshow("Image1", sourcea[0])
    # cv2.imshow("Image2", sourceb[0])  
    # cv2.imshow("Image2 realigned", img2_reg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Affichage des 2 images superposées
    # overlay = sourcea[0].copy()
    # cv2.addWeighted(sourcea[0], 0.5, img2_reg, 0.5, 0, overlay)
    # cv2.imshow("Images overlaid after the first alignment", overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    color1 = sourcea[0].copy()
    color2 = img2_reg
    gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)   # On convertit en niveaux de gris
    gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
    _, gray1_binarised = cv2.threshold(gray1, gray_threshold, 255, cv2.THRESH_BINARY)
    _, gray2_binarised = cv2.threshold(gray2, gray_threshold, 255, cv2.THRESH_BINARY)

    # Affichage des 2 images binarisées superposées
    overlay = np.zeros_like(gray1_binarised)
    cv2.addWeighted(gray1_binarised, 0.5, gray2_binarised, 0.5, 0, overlay)
    cv2.imshow("Images overlaid after the first alignment", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    # Calcul de la différence entre les 2 images (image 1 originale et image 2 transformée)
    color1 = sourcea[0].copy()
    color2 = img2_reg
    if print_graphs:
        print("Affichage des colors")
        show2_with_cursor(color1, color2)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)   # On convertit en niveaux de gris
    gray2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
    if print_graphs:
        print("Affichage des grays")
        show2_with_cursor(gray1, gray2)
    _, gray1 = cv2.threshold(gray1, gray_threshold, 255, cv2.THRESH_BINARY)
    _, gray2 = cv2.threshold(gray2, gray_threshold, 255, cv2.THRESH_BINARY)
    if print_graphs:
        print("Affichage des grays binarisés")
        show2_with_cursor(gray1, gray2)
    h = min(gray1.shape[0], gray2.shape[0])
    w = min(gray1.shape[1], gray2.shape[1])
    # gray1 = cv2.resize(gray1, shape_of_diff, interpolation=cv2.INTER_AREA)
    # gray2 = cv2.resize(gray2, shape_of_diff, interpolation=cv2.INTER_AREA)
    gray1 = minkowski_resize(gray1, shape_of_diff, minkowski_mean_order)
    gray2 = minkowski_resize(gray2, shape_of_diff, minkowski_mean_order)
    if print_graphs:
        print("Affichage des grays redimensionnés")
        show2_with_cursor(gray1, gray2)

    diff = cv2.absdiff(gray1, gray2)
    diff_cut = (diff>diff_threshold) * diff
    answer = np.all(diff<=diff_threshold)
    print("Les 2 feuilles de papier ", sourcea[1]," et ", sourceb[1], " sont identiques :", answer)
    if print_graphs:
        show2_with_cursor(diff_cut, diff)
    
    return answer.item()
    '''

    ## --------- 2è alignement -------------
    img1_2 = gray1_binarised.copy()
    img2_2 = gray2_binarised.copy()

    # Détection des keypoints
    sift = cv2.SIFT_create(nfeatures = 3000)
    keypoints1, descriptors1 = sift.detectAndCompute(img1_2, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_2, None)
    img1_2_with_keypoints = cv2.drawKeypoints(img1_2, keypoints1, None, color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Image 1 - keypoints pour second alignement", img1_2_with_keypoints)
    img2_2_with_keypoints = cv2.drawKeypoints(img2_2, keypoints2, None, color = (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Image 2 - keypoints pour second alignement", img2_2_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Matching des keypoints
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    all_matches = list(matcher.knnMatch(descriptors1, descriptors2, k=2))
    # Appliquer le test de ratio de Lowe
    good_matches = []
    for m, n in all_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    numGoodMatches = int(len(good_matches) * topMatchesFactor2)
    matches = good_matches[:numGoodMatches]
    im_matches = cv2.drawMatches(img1_2, keypoints1, img2_2, keypoints2, matches, None)
    cv2.imshow("Image matches", im_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calcul de la transformation
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, _ = sourcea[0].shape
    img2_2_reg = cv2.warpPerspective(img2_2, h, (width, height))
    cv2.imshow("Image1", gray1_binarised)
    cv2.imshow("Image2 realigned once", gray2_binarised)  
    cv2.imshow("Image2 realigned twice", img2_2_reg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Affichage des 2 images superposées
    overlay2 = np.zeros_like(img1_2)
    cv2.addWeighted(img1_2, 0.5, img2_2_reg, 0.5, 0, overlay2)
    cv2.imshow("Images overlaid - first alignment", overlay)
    cv2.imshow("Images overlaid - second alignment", overlay2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Récolte de toutes les images de test
test = [[cv2.imread(os.path.join(BASE_DIR, "./test_set", "IMG_1702.jpeg")), "IMG_1702.jpeg"]]
for filename in os.listdir("/Users/noahparisse/Documents/Paris-digital-lab/P1 RTE/detection-notes/src/recog/test_set"):
    if filename == "IMG_1702.jpeg":
        continue
    test.append([cv2.imread(os.path.join(BASE_DIR, "./test_set", filename)), filename])

# Test du process sur toutes les images sélectionnées
for i, img in enumerate(test):
    if i == 0:
        continue
    # if i==6:
    print("Test de l'image", img[1])
    double_alignment(test[0], img)