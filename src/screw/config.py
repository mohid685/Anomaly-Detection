import os
import torch

# Project root = two levels above this file (src/screw/.. -> src -> ..)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "screw")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train", "good")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
GT_DIR = os.path.join(DATASET_ROOT, "ground_truth")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "screw")

# Image settings
IMAGE_SIZE = 256      # resize dimension
CENTER_CROP = 224     # crop fed to backbone

# PatchCore settings
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer1", "layer2", "layer3"]     # feature layers to extract
CORESET_RATIO = 0.25                        # starting point — same as metal_nut's final config
NUM_NEIGHBORS = 9                           # k in k-NN scoring

# Device
DEVICE = "cuda" if (torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1") else "cpu"

# Output
MEMORY_BANK_PATH = os.path.join(ARTIFACTS_DIR, "memory_bank.pt")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")