import os
import sys
import json
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

# Reuse your existing, already-validated pipeline code directly
COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common")
sys.path.insert(0, COMMON_DIR)

from patchcore import PatchCore
from mvtec_dataset import get_transform

CATEGORY_CONFIG = {
    "metal_nut": {"top_k": 5, "use_position_norm": False, "subtract_baseline": False},
    "screw": {"top_k": 20, "use_position_norm": True, "subtract_baseline": False},
    "cable": {"top_k": 5, "use_position_norm": True, "subtract_baseline": False},
    "transistor": {"top_k": 5, "use_position_norm": True, "subtract_baseline": False},
}

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARTIFACTS_ROOT = os.path.join(PROJECT_ROOT, "artifacts")

_models = {}
_thresholds = {}
_transform = get_transform()


def load_all_models():
    """Loads every category's memory bank once at server startup."""
    for category in CATEGORY_CONFIG:
        memory_bank_path = os.path.join(ARTIFACTS_ROOT, category, "memory_bank.pt")
        threshold_path = os.path.join(ARTIFACTS_ROOT, category, "threshold.json")

        if not os.path.exists(memory_bank_path):
            print(f"WARNING: no memory bank found for '{category}' at {memory_bank_path} — skipping.")
            continue

        model = PatchCore()
        model.load(memory_bank_path)
        _models[category] = model

        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                _thresholds[category] = json.load(f)["threshold"]
        else:
            _thresholds[category] = 0.0

        print(f"Loaded model for category: {category}")

    if not _models:
        raise RuntimeError("No trained models found under artifacts/. Train at least one category first.")


def available_categories():
    return list(_models.keys())


def _encode_heatmap_overlay(image_tensor, anomaly_map):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    size = (img_uint8.shape[1], img_uint8.shape[0])
    heatmap = cv2.resize(anomaly_map, size, interpolation=cv2.INTER_CUBIC)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (0.55 * img_uint8 + 0.45 * heatmap_color).astype(np.uint8)

    pil_img = Image.fromarray(overlay)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(category, pil_image):
    if category not in _models:
        raise ValueError(f"No trained model available for category '{category}'.")

    model = _models[category]
    threshold = _thresholds[category]
    cfg = CATEGORY_CONFIG[category]

    tensor = _transform(pil_image.convert("RGB")).unsqueeze(0)

    score, anomaly_map = model.score(
        tensor,
        top_k=cfg["top_k"],
        subtract_baseline=cfg["subtract_baseline"],
        use_position_norm=cfg["use_position_norm"],
    )

    is_defective = bool(score > threshold)
    heatmap_b64 = _encode_heatmap_overlay(tensor[0], anomaly_map)

    return {
        "category": category,
        "score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "is_defective": is_defective,
        "heatmap_base64": heatmap_b64,
    }