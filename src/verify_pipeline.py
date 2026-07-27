"""
Lightweight smoke test for the metal_nut / screw PatchCore pipelines.
Verifies imports resolve correctly and runs a tiny end-to-end pass
(a handful of images) WITHOUT doing a full training run.

Run from src/:
    python verify_pipeline.py metal_nut
    python verify_pipeline.py screw
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


def setup_paths(category):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    category_dir = os.path.join(src_dir, category)
    common_dir = os.path.join(src_dir, "common")

    if not os.path.isdir(category_dir):
        raise SystemExit(f"No such category folder: {category_dir}")

    for path in (category_dir, common_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("category", choices=["metal_nut", "screw"])
    args = parser.parse_args()

    print(f"=== Verifying pipeline for: {args.category} ===\n")
    setup_paths(args.category)

    # Clear any previously cached modules from a prior category run in the
    # same interpreter session (relevant if this script is ever imported
    # rather than run standalone).
    for mod in ["config", "mvtec_dataset", "patchcore"]:
        sys.modules.pop(mod, None)

    print("[1/6] Importing config...")
    import config
    print(f"      DATASET_ROOT = {config.DATASET_ROOT}")
    print(f"      ARTIFACTS_DIR = {config.ARTIFACTS_DIR}")
    print(f"      DEVICE = {config.DEVICE}")

    print("[2/6] Checking dataset paths exist...")
    for name, path in [("TRAIN_DIR", config.TRAIN_DIR), ("TEST_DIR", config.TEST_DIR), ("GT_DIR", config.GT_DIR)]:
        ok = os.path.isdir(path)
        print(f"      {name}: {path} -> {'OK' if ok else 'MISSING'}")
        if not ok:
            raise SystemExit(f"Required path missing: {path}")

    print("[3/6] Importing mvtec_dataset and patchcore...")
    from mvtec_dataset import MVTecTrainDataset, MVTecTestDataset
    from patchcore import PatchCore
    print("      OK")

    print("[4/6] Loading a small slice of train/test data...")
    train_dataset = MVTecTrainDataset(config.TRAIN_DIR)
    test_dataset = MVTecTestDataset(config.TEST_DIR, config.GT_DIR)
    print(f"      Full train set size: {len(train_dataset)}")
    print(f"      Full test set size:  {len(test_dataset)}")

    # Build a tiny 5-image subset for a fast smoke test
    small_imgs = torch.stack([train_dataset[i] for i in range(min(5, len(train_dataset)))])
    small_loader = [small_imgs]  # fake a single-batch "loader"

    print("[5/6] Running a tiny fit() + score() pass (not a real training run)...")
    model = PatchCore()
    model.fit(small_loader)

    correct_shapes = True
    for i in range(min(3, len(test_dataset))):
        img, label, mask, defect_type, path = test_dataset[i]
        score, anomaly_map = model.score(img.unsqueeze(0))
        if not np.isfinite(score) or anomaly_map.ndim != 2:
            correct_shapes = False
        print(f"      sample[{i}] defect_type={defect_type} label={label} score={score:.4f}")

    if not correct_shapes:
        raise SystemExit("Smoke test FAILED: invalid score or anomaly map shape.")

    print("[6/6] Verifying save/load round-trip on a temp path...")
    tmp_path = os.path.join(config.ARTIFACTS_DIR, "verify_tmp_memory_bank.pt")
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    model.save(tmp_path)
    reloaded = PatchCore()
    reloaded.load(tmp_path)
    assert np.allclose(reloaded.memory_bank, model.memory_bank)
    os.remove(tmp_path)
    print("      OK")

    print(f"\n=== {args.category} pipeline verified successfully. Safe to run full training. ===")


if __name__ == "__main__":
    main()