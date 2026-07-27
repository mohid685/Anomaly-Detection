import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import csv
import json
from torch.utils.data import DataLoader

from config import TEST_DIR, GT_DIR, MEMORY_BANK_PATH, ARTIFACTS_DIR, THRESHOLD_PATH
from common.mvtec_dataset import MVTecTestDataset
from common.patchcore import PatchCore
from common.visualize import overlay_heatmap


def main():
    test_dataset = MVTecTestDataset(TEST_DIR, GT_DIR)

    model = PatchCore()
    model.load(MEMORY_BANK_PATH)

    threshold = None
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            threshold = json.load(f)["threshold"]

    output_dir = os.path.join(ARTIFACTS_DIR, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(ARTIFACTS_DIR, "scores_summary.csv")
    rows = []

    for img_t, label, mask, defect_type, path in test_dataset:
        fname = os.path.splitext(os.path.basename(path))[0]

        type_dir = os.path.join(output_dir, defect_type)
        os.makedirs(type_dir, exist_ok=True)

        image_score, anomaly_map = model.score(img_t.unsqueeze(0))

        save_path = os.path.join(type_dir, f"{fname}.png")
        overlay_heatmap(img_t, anomaly_map, save_path=save_path)

        predicted_defective = None
        correct = None
        if threshold is not None:
            predicted_defective = bool(image_score > threshold)
            correct = (predicted_defective == bool(label))

        rows.append({
            "defect_type": defect_type,
            "filename": fname,
            "true_label": label,
            "image_score": round(float(image_score), 4),
            "threshold": round(threshold, 4) if threshold is not None else "",
            "predicted_defective": predicted_defective,
            "correct": correct,
            "visualization_path": save_path,
        })

        print(f"[{defect_type}] {fname}: score={image_score:.4f} "
              f"{'CORRECT' if correct else ('WRONG' if correct is not None else '')}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Misclassification summary ===")
    by_type = {}
    for r in rows:
        by_type.setdefault(r["defect_type"], {"total": 0, "wrong": 0})
        by_type[r["defect_type"]]["total"] += 1
        if r["correct"] is False:
            by_type[r["defect_type"]]["wrong"] += 1

    for defect_type, stats in by_type.items():
        print(f"  {defect_type}: {stats['wrong']}/{stats['total']} misclassified")

    print(f"\nFull per-image results saved to: {csv_path}")
    print(f"All heatmap visualizations saved under: {output_dir}")


if __name__ == "__main__":
    main()