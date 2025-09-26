import cv2
from shape_detector import shape_detector
from save_detection import save_detection
from image_preprocessing import preprocessed_image

video_path = "src/proc/paper_detection/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Impossible de lire la vidéo : {video_path}")

# Préparer sauvegarde de la vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_detected.mp4", fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, img = cap.read()
    if not ret:
        print("✅ Fin de la vidéo")
        break

    # Détection
    possible_papers = shape_detector(img)
    img_show = img.copy()

    # Colorier chaque quadrilatère
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    for i, quad in enumerate(possible_papers):
        color = colors[i % len(colors)]
        cv2.drawContours(img_show, [quad], -1, color, 3)
        for point in quad:
            x, y = point[0]
            cv2.circle(img_show, (x, y), 6, (255, 0, 255), -1)

    # Infos en overlay
    cv2.putText(
        img_show,
        f"Feuilles detectees: {len(possible_papers)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if len(possible_papers) > 0 else (0, 0, 255),
        2
    )

    # Screenshot lorsqu'il y a détection
    if len(possible_papers) > 0:
        save_detection(img, possible_papers)

    # Affichage + sauvegarde
    cv2.imshow("Video", img_show)
    out.write(img_show)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
