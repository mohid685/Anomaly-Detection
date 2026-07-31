import os
import sys

CABLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON_DIR = os.path.join(os.path.dirname(CABLE_DIR), "common")

for path in (CABLE_DIR, COMMON_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)