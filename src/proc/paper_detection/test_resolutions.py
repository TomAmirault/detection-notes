import cv2
import time

resolutions = [
    (320, 240), (640, 480), (800, 600), (1024, 768),
    (1280, 720), (1280, 800), (1600, 1200), (1920, 1080),
    (2560, 1440), (3840, 2160), (1920, 1552), (1920, 1328)
]

cap = cv2.VideoCapture(0)

print("🔍 Test des résolutions supportées :")
for w, h in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(0.2)
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if abs(real_w - w) < 5 and abs(real_h - h) < 5:
        print(f"✅ Supportée : {int(real_w)}x{int(real_h)}")
    else:
        print(f"❌ Demandée {w}x{h} -> obtenu {int(real_w)}x{int(real_h)}")

cap.release()