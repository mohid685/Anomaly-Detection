import os
import sys

# Make the shared 'common' package importable regardless of where this script is run from
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

from torch.utils.data import DataLoader

from config import TRAIN_DIR, ARTIFACTS_DIR, MEMORY_BANK_PATH
from mvtec_dataset import MVTecTrainDataset
from patchcore import PatchCore


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    train_dataset = MVTecTrainDataset(TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    print(f"Loaded {len(train_dataset)} defect-free training images.")

    model = PatchCore()
    model.fit(train_loader)
    model.save(MEMORY_BANK_PATH)

    print(f"Memory bank saved to {MEMORY_BANK_PATH}")


if __name__ == "__main__":
    main()