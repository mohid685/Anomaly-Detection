import os
import sys

# metal_nut/tests -> metal_nut (has config.py)
METAL_NUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# metal_nut/tests -> metal_nut -> src -> src/common (has patchcore.py, mvtec_dataset.py)
COMMON_DIR = os.path.join(os.path.dirname(METAL_NUT_DIR), "common")

for path in (METAL_NUT_DIR, COMMON_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)