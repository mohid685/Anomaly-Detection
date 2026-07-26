import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, CENTER_CROP


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(CENTER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def get_mask_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(CENTER_CROP),
        transforms.ToTensor(),
    ])


class MVTecTrainDataset(Dataset):
    """Loads only defect-free training images (train/good)."""

    def __init__(self, train_dir):
        self.paths = sorted(
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.lower().endswith(".png")
        )
        self.transform = get_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class MVTecTestDataset(Dataset):
    """
    Loads all test images across every defect subfolder ('good' included).
    Returns image, binary label (0=good, 1=defect), mask, and defect type.
    """

    def __init__(self, test_dir, gt_dir):
        self.samples = []  # (image_path, label, mask_path_or_None, defect_type)
        self.transform = get_transform()
        self.mask_transform = get_mask_transform()

        for defect_type in sorted(os.listdir(test_dir)):
            sub_dir = os.path.join(test_dir, defect_type)
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                img_path = os.path.join(sub_dir, fname)
                if defect_type == "good":
                    self.samples.append((img_path, 0, None, defect_type))
                else:
                    mask_name = fname.replace(".png", "_mask.png")
                    mask_path = os.path.join(gt_dir, defect_type, mask_name)
                    self.samples.append((img_path, 1, mask_path, defect_type))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path, defect_type = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask_t = self.mask_transform(mask)
        else:
            mask_t = torch.zeros(1, CENTER_CROP, CENTER_CROP)  # placeholder for good images

        return img_t, label, mask_t, defect_type, img_path

    