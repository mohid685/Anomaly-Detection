import os
import sys

# Add the parent 'src' directory to sys.path so tests can import
# config, patchcore, mvtec_dataset, etc. directly, the same way the
# scripts in src/ do.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)