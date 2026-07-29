import os
import sys

# screw/tests -> screw (has config.py)
SCREW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# screw/tests -> screw -> src -> src/common (has patchcore.py, mvtec_dataset.py)
COMMON_DIR = os.path.join(os.path.dirname(SCREW_DIR), "common")

for path in (SCREW_DIR, COMMON_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)