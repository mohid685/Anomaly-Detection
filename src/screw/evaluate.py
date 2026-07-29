import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import json
import numpy as np
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from config import (
    TEST_DIR, GT_DIR, MEMORY_BANK_PATH, THRESHOLD_PATH, CENTER_CROP,
    TOP_K, SUBTRACT_BASELINE, USE_POSITION_NORM,
)
from mvtec_dataset import MVTecTestDataset
from patchcore import PatchCore


def upsample_map(anomaly_map, size):
    return cv2.resize(anomaly_map, size, interpolation=cv2.INTER_CUBIC)


def main():
    test_dataset = MVTecTestDataset(TEST_DIR, GT_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = PatchCore()
    model.load(MEMORY_BANK_PATH)

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    per_type_results = {}

    for img, label, mask, defect_type, path in test_loader:
        label = int(label.item())
        defect_type = defect_type[0]

        image_score, anomaly_map = model.score(
            img, top_k=TOP_K, subtract_baseline=SUBTRACT_BASELINE, use_position_norm=USE_POSITION_NORM
        )
        image_scores.append(image_score)
        image_labels.append(label)

        upsampled = upsample_map(anomaly_map, (CENTER_CROP, CENTER_CROP))

        if mask is not None and mask.numel() > 0:
            gt = (mask.squeeze().numpy() > 0.5).astype(np.uint8)
            pixel_scores.append(upsampled.flatten())
            pixel_labels.append(gt.flatten())
        elif label == 0:
            gt = np.zeros((CENTER_CROP, CENTER_CROP), dtype=np.uint8)
            pixel_scores.append(upsampled.flatten())
            pixel_labels.append(gt.flatten())

        per_type_results.setdefault(defect_type, []).append(image_score)

    image_auroc = roc_auc_score(image_labels, image_scores)
    pixel_auroc = roc_auc_score(
        np.concatenate(pixel_labels), np.concatenate(pixel_scores)
    )

    print(f"Image-level AUROC: {image_auroc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")

    for defect_type, scores in per_type_results.items():
        print(f"  {defect_type}: mean score {np.mean(scores):.4f}")

    fpr, tpr, thresholds = roc_curve(image_labels, image_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    threshold = float(thresholds[best_idx])

    print(f"\nOptimal threshold (Youden's J): {threshold:.4f}")
    print(f"  At this threshold -> TPR: {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({
            "threshold": threshold,
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "tpr_at_threshold": float(tpr[best_idx]),
            "fpr_at_threshold": float(fpr[best_idx]),
        }, f, indent=2)

    print(f"Threshold saved to {THRESHOLD_PATH}: {threshold:.4f}")


if __name__ == "__main__":
    main()