# Ce fichier définit les fonctions utiles à la recherche de la frame la moins floue provenant d'une vidéo
import cv2
import time

# Cette fonction renvoie la variance du laplacien de la matrice img représentant les niveaux de gris de
# l'image sur laquelle on souhaite travailler. Plus un eimage est floue, et plus la variance de son laplacien
# sera petite.
def laplacian_variance(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = cv2.Laplacian(res, cv2.CV_64F, ksize = 3).var()
    return value

# Cette fonction renvoie l'indice de l'image la moins floue de video en utilisant laplacian_variance
def less_blurred(video):    # video = une liste d'images
    start = time.time()
    best = 0
    best_value = laplacian_variance(video[0].orig_img)
    for i in range(1,len(video)):
        value_tmp = laplacian_variance(video[i].orig_img)
        if value_tmp>best_value:
            best_value=value_tmp
            best = i
    print(f"Temps d'exécution de less_blurred : {time.time()-start} secondes")
    return best