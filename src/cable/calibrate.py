import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

from torch.utils.data import DataLoader
from config import TRAIN_DIR, MEMORY_BANK_PATH
from mvtec_dataset import MVTecTrainDataset
from patchcore import PatchCore


def main():
    train_dataset = MVTecTrainDataset(TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    model = PatchCore()
    model.load(MEMORY_BANK_PATH)
    model.calibrate(train_loader)
    model.save(MEMORY_BANK_PATH)

    print("Position calibration saved into memory_bank.pt")


if __name__ == "__main__":
    main()