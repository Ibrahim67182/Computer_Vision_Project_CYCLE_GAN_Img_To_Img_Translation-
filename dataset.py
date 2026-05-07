"""
CycleGAN - Dataset Loader
Handles image loading, resizing, normalization for all three datasets.

Usage example:
    from dataset import get_loader

    train_loader = get_loader(dataset="horse2zebra", mode="train", batch_size=1)
    test_loader  = get_loader(dataset="horse2zebra", mode="test",  batch_size=1)
"""


# for image normalization and augmentation of 256x256 

# In short — it loads images, transforms them on the fly, and feeds them to the model in random unpaired batches. Nothing is saved to disk, raw images stay untouched.

import os
import glob
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────

def get_transforms(mode="train", image_size=256):
    """
    Returns torchvision transforms for train or test mode.

    Train:
        - Resize to 286x286 (slightly larger)
        - Random crop to 256x256  (standard CycleGAN augmentation)
        - Random horizontal flip
        - Convert to tensor
        - Normalize to [-1, 1]  (matches Tanh output of generator)

    Test:
        - Resize directly to 256x256
        - Convert to tensor
        - Normalize to [-1, 1]
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.12), Image.BICUBIC),  # 286x286
            transforms.RandomCrop(image_size),                          # 256x256
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
    else:  # test
        return transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])


# ─────────────────────────────────────────────
#  Dataset Class
# ─────────────────────────────────────────────

class UnpairedDataset(Dataset):
    """
    Loads images from domainA and domainB independently (unpaired).
    Each __getitem__ returns one random image from A and one from B.
    The two images do NOT correspond to each other — that is intentional.

    Folder structure expected:
        datasets/
            horse2zebra/
                trainA/  trainB/  testA/  testB/

    Args:
        root        : path to the datasets/ folder
        dataset     : one of "horse2zebra", "monet2photo", "summer2winter"
        mode        : "train" or "test"
        image_size  : resize target (default 256)
    """

    def __init__(self, root, dataset, mode="train", image_size=256):
        self.transform = get_transforms(mode, image_size)

        dir_A = os.path.join(root, dataset, f"{mode}A")
        dir_B = os.path.join(root, dataset, f"{mode}B")

        # Collect all image paths (jpg and png)
        self.files_A = sorted(
            glob.glob(os.path.join(dir_A, "*.jpg")) +
            glob.glob(os.path.join(dir_A, "*.png")) +
            glob.glob(os.path.join(dir_A, "*.jpeg"))
        )
        self.files_B = sorted(
            glob.glob(os.path.join(dir_B, "*.jpg")) +
            glob.glob(os.path.join(dir_B, "*.png")) +
            glob.glob(os.path.join(dir_B, "*.jpeg"))
        )

        if len(self.files_A) == 0:
            raise FileNotFoundError(f"No images found in: {dir_A}")
        if len(self.files_B) == 0:
            raise FileNotFoundError(f"No images found in: {dir_B}")

        print(f"[{dataset} | {mode}]  Domain A: {len(self.files_A)} images  |  Domain B: {len(self.files_B)} images")

    def __len__(self):
        # Length is the larger of the two domains
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        # Wrap index so neither domain runs out
        img_A = self._load(self.files_A[index % len(self.files_A)])
        img_B = self._load(self.files_B[random.randint(0, len(self.files_B) - 1)])
        return {"A": img_A, "B": img_B}

    def _load(self, path):
        img = Image.open(path).convert("RGB")  # ensure 3-channel (drops alpha if PNG)
        return self.transform(img)


# ─────────────────────────────────────────────
#  Convenience Loader Function
# ─────────────────────────────────────────────

def get_loader(dataset, mode="train", root="datasets", batch_size=1,
               image_size=256, num_workers=2, shuffle=None):
    """
    Returns a DataLoader for the given dataset and mode.

    Args:
        dataset     : "horse2zebra" | "monet2photo" | "summer2winter"
        mode        : "train" | "test"
        root        : path to datasets/ folder (default: "datasets")
        batch_size  : images per batch (default 1, as in the paper)
        image_size  : 256
        num_workers : parallel workers for loading
        shuffle     : True for train, False for test (auto if None)

    Returns:
        torch.utils.data.DataLoader
    """
    if shuffle is None:
        shuffle = (mode == "train")

    ds = UnpairedDataset(root=root, dataset=dataset, mode=mode, image_size=image_size)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,   # faster GPU transfer
    )


# ─────────────────────────────────────────────
#  Quick Verification (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt

    DATASET   = "horse2zebra"   # change to test other datasets
    ROOT      = "datasets"

    print("\n--- Checking train loader ---")
    train_loader = get_loader(DATASET, mode="train", root=ROOT, batch_size=1)

    print("--- Checking test loader ---")
    test_loader = get_loader(DATASET, mode="test", root=ROOT, batch_size=1)

    # Grab one batch
    batch = next(iter(train_loader))
    img_A = batch["A"]  # shape: [1, 3, 256, 256]
    img_B = batch["B"]

    print(f"\nBatch A shape : {img_A.shape}")
    print(f"Batch B shape : {img_B.shape}")
    print(f"Pixel min/max : {img_A.min():.2f} / {img_A.max():.2f}  (should be near -1 and 1)")

    # Denormalize helper for visualization
    def denorm(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1).squeeze().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(denorm(img_A)); axes[0].set_title("Domain A sample"); axes[0].axis("off")
    axes[1].imshow(denorm(img_B)); axes[1].set_title("Domain B sample"); axes[1].axis("off")
    plt.suptitle(f"Dataset: {DATASET}")
    plt.tight_layout()
    plt.savefig("dataset_check.png", dpi=120)
    print("\nSample images saved to dataset_check.png")
    plt.show()