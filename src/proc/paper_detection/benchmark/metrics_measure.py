import os
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

iou_threshold = 0.5

def load_mask(path):
    """Charge un masque (image ou .npy) et le binarise."""
    if path.endswith(".npy"):
        mask = np.load(path)
    else:
        print("Le path ne finit pas par .npy")
    return mask

def evaluate_folder(gt_dir, pred_dir, iou_threshold):
    ious, dices, precisions, recalls, accs, detections = [], [], [], [], [], []
    files = [f for f in os.listdir(gt_dir) if f.endswith((".png", ".jpg", ".npy"))]

    for fname in files:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            print(f"⚠️ Pas de prédiction pour {fname}")
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            pred = np.array(Image.fromarray(pred).resize(gt.shape[::-1], resample=Image.NEAREST))

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()

        iou = jaccard_score(gt_flat, pred_flat)
        ious.append(iou)
        dices.append(f1_score(gt_flat, pred_flat))
        precisions.append(precision_score(gt_flat, pred_flat))
        recalls.append(recall_score(gt_flat, pred_flat))
        if iou>iou_threshold:
            detections.append(1)
        else :
            detections.append(0)
        accs.append(accuracy_score(gt_flat, pred_flat))
    detection_recall = np.sum(detections)/len(detections)
    return {
        "mIoU": np.mean(ious),
        "Correct images": detection_recall,
        "mean Dice": np.mean(dices),
        "Mean pixel Precision": np.mean(precisions),
        "Mean pixel Recall": np.mean(recalls),
        "Mean pixel Accuracy": np.mean(accs),
        "N images": len(ious)
    }




gt_dir = os.path.join(BASE_DIR, "./data/set/gt_labels")
pred_yoloseg_dir = os.path.join(BASE_DIR, "./data/yolo-seg")
pred_yolodetect_dir = os.path.join(BASE_DIR, "./data/yolo-detect")
pred_alex_dir = os.path.join(BASE_DIR, "./data/alex-model")
pred_moha_dir = os.path.join(BASE_DIR, "./data/yolo-seg-moha")

print("==> YOLO-seg")
metrics_yoloseg = evaluate_folder(gt_dir, pred_yoloseg_dir, iou_threshold)
for k, v in metrics_yoloseg.items():
    print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("==> YOLO-detect + grabcut")
metrics_yolodetect = evaluate_folder(gt_dir, pred_yolodetect_dir, iou_threshold)
for k, v in metrics_yolodetect.items():
    print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("==> YOLO-seg-moha")
metrics_yolosegmoha = evaluate_folder(gt_dir, pred_moha_dir, iou_threshold)
for k, v in metrics_yolosegmoha.items():
    print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n==> Modèle d'Alex")
metrics_alex = evaluate_folder(gt_dir, pred_alex_dir, iou_threshold)
for k, v in metrics_alex.items():
    print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Comparaison rapide
print("\n📊 Comparaison")
for k in ["mIoU", "mean Dice", "Correct images", "Mean pixel Precision", "Mean pixel Recall", "Mean pixel Accuracy"]:
    print(f"{k:>20}: YOLO-seg = {metrics_yoloseg[k]:.4f} | YOLO-detect = {metrics_yolodetect[k]:.4f} | YOLO-seg-moha = {metrics_yolosegmoha[k]:.4f} | Modèle d'Alex = {metrics_alex[k]:.4f}")