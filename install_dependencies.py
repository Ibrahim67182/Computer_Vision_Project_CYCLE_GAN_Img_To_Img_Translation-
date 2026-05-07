"""
CycleGAN Project - Dependency Installer
Run this once in the lab before starting the project.
Usage: python install_dependencies.py
"""

# for importing all important libraries and installing missing ones at once s

import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARNING] Command failed: {cmd}")
    else:
        print("[OK]")

print("=" * 50)
print("  CycleGAN - Installing Dependencies")
print("=" * 50)

# Upgrade pip first
run(f"{sys.executable} -m pip install --upgrade pip")

# Core deep learning
run(f"{sys.executable} -m pip install torch torchvision torchaudio")

# Image processing
run(f"{sys.executable} -m pip install Pillow numpy")

# Visualization
run(f"{sys.executable} -m pip install matplotlib")

# Training progress bar
run(f"{sys.executable} -m pip install tqdm")

# Optional but useful
run(f"{sys.executable} -m pip install tensorboard")   # loss curve logging
run(f"{sys.executable} -m pip install kaggle")        # to download datasets via CLI

print("\n" + "=" * 50)
print("  All done! Verifying installs...")
print("=" * 50)

# Verification
libs = [
    ("torch",       "PyTorch"),
    ("torchvision", "Torchvision"),
    ("PIL",         "Pillow"),
    ("numpy",       "NumPy"),
    ("matplotlib",  "Matplotlib"),
    ("tqdm",        "tqdm"),
]

all_ok = True
for module, name in libs:
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [MISSING] {name} - install failed, check above logs")
        all_ok = False

# Check CUDA
try:
    import torch
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "N/A"
    print(f"\n  CUDA available : {cuda}")
    print(f"  GPU            : {gpu_name}")
    if not cuda:
        print("  [WARNING] No GPU detected - training will be very slow on CPU")
except Exception as e:
    print(f"  [ERROR] Could not check CUDA: {e}")

print("\n" + "=" * 50)
if all_ok:
    print("  Setup complete. You're ready to train!")
else:
    print("  Some packages failed. Check the warnings above.")
print("=" * 50)