import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import CENTER_CROP


def overlay_heatmap(image_tensor, anomaly_map, save_path=None):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)

    heatmap = cv2.resize(anomaly_map, (CENTER_CROP, CENTER_CROP))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Anomaly Map"); axes[1].axis("off")
    axes[2].imshow(img); axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay"); axes[2].axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)