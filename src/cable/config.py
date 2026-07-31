import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "cable")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train", "good")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
GT_DIR = os.path.join(DATASET_ROOT, "ground_truth")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "cable")

IMAGE_SIZE = 256
CENTER_CROP = 224

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.25
NUM_NEIGHBORS = 9

# Scoring strategy — position norm ON by default here, since cable is
# photographed in a fixed pose (unlike screw's free rotation), so per-location
# calibration should be reliable from the first run rather than a later fix.
TOP_K = 2
SUBTRACT_BASELINE = False
USE_POSITION_NORM = True

DEVICE = "cuda" if (torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1") else "cpu"

MEMORY_BANK_PATH = os.path.join(ARTIFACTS_DIR, "memory_bank.pt")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")