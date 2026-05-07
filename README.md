# CycleGAN — Unpaired Image-to-Image Translation

> Transform images between domains without paired training data. Horses become zebras, photos become Monet paintings, summer landscapes turn to winter — and back again.

---

## Table of Contents

- [What is CycleGAN?](#what-is-cyclegan)
- [Pre-Trained Models](#pre-trained-models)
- [Installation](#installation)
- [Running Inference (predict.py)](#running-inference-predictpy)
- [Training Your Own Model (train.py)](#training-your-own-model-trainpy)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Example Outputs](#example-outputs)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## What is CycleGAN?

CycleGAN performs **unpaired image-to-image translation** — meaning it learns to translate images between two domains (e.g. horses and zebras) without ever needing matched image pairs.

It works using **cycle consistency**: if you translate a horse to a zebra and then translate that zebra back, you should get the original horse. This self-supervision is what makes CycleGAN powerful with unstructured, unpaired datasets.

```
Horse → Generator(A→B) → Fake Zebra → Generator(B→A) → Reconstructed Horse ≈ Original
```

**Original Paper:** _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros — ICCV 2017
[ArXiv: 1703.10593](https://arxiv.org/abs/1703.10593)

---

## Pre-Trained Models

Three fully trained models are included in the `checkpoints/` folder, ready for immediate inference:

| Model           | Domain A         | Domain B         | Trained Epochs |
| --------------- | ---------------- | ---------------- | -------------- |
| `horse2zebra`   | Horse            | Zebra            | 300            |
| `summer2winter` | Summer landscape | Winter landscape | 250            |
| `monet2photo`   | Monet painting   | Real photograph  | 400            |

---

## Installation

### Requirements

- Python 3.8+
- 8 GB RAM (16 GB recommended)
- GPU with 4 GB+ VRAM (recommended) — CPU fallback supported

### Step 1 — Clone the repo

```bash
git clone https://github.com/Ibrahim67182/Computer_Vision_Project_CYCLE_GAN_Img_To_Img_Translation-.git
cd Computer_Vision_Project_CYCLE_GAN_Img_To_Img_Translation-
```

### Step 2 — Install dependencies

```bash
python install_dependencies.py
```

Or manually:

```bash
pip install torch torchvision torchaudio Pillow numpy matplotlib tqdm
```

### Step 3 — Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Running Inference (predict.py)

Use `predict.py` to translate any image using the pre-trained models. The output will be saved to the path you specify (e.g. `results/your_output_name.png`).

### Parameters

| Parameter      | Required | Description                                                        |
| -------------- | -------- | ------------------------------------------------------------------ |
| `--input`      | Yes      | Path to your input image                                           |
| `--checkpoint` | Yes      | Path to the `.pth` model weights file                              |
| `--output`     | Yes      | Where to save the output image                                     |
| `--mode`       | Yes      | Translation mode: `horse2zebra`, `summer2winter`, or `monet2photo` |

---

### Horse ↔ Zebra

_Domain A = Horse, Domain B = Zebra_

**Horse → Zebra:**

```bash
python predict.py \
  --input <path_to_horse_image> \
  --checkpoint checkpoints/horse2zebra/G_AB_epoch_300.pth \
  --output results/<your_output_image_name>.png \
  --mode horse2zebra
```

**Zebra → Horse:**

```bash
python predict.py \
  --input <path_to_zebra_image> \
  --checkpoint checkpoints/horse2zebra/G_BA_epoch_300.pth \
  --output results/<your_output_image_name>.png \
  --mode horse2zebra
```

---

### Summer ↔ Winter

_Domain A = Summer, Domain B = Winter_

**Summer → Winter:**

```bash
python predict.py \
  --input <path_to_summer_image> \
  --checkpoint checkpoints/summer2winter/G_AB_epoch_250.pth \
  --output results/<your_output_image_name>.png \
  --mode summer2winter
```

**Winter → Summer:**

```bash
python predict.py \
  --input <path_to_winter_image> \
  --checkpoint checkpoints/summer2winter/G_BA_epoch_250.pth \
  --output results/<your_output_image_name>.png \
  --mode summer2winter
```

---

### Monet ↔ Photo

_Domain A = Monet painting, Domain B = Real photograph_

**Monet → Photo:**

```bash
python predict.py \
  --input <path_to_monet_painting> \
  --checkpoint checkpoints/monet2photo/G_AB_epoch_400.pth \
  --output results/<your_output_image_name>.png \
  --mode monet2photo
```

**Photo → Monet:**

```bash
python predict.py \
  --input <path_to_real_photo> \
  --checkpoint checkpoints/monet2photo/G_BA_epoch_400.pth \
  --output results/<your_output_image_name>.png \
  --mode monet2photo
```

> **Tips for best results:**
>
> - Input images between 256×256 and 1024×1024 work best
> - JPG, PNG, BMP, and TIFF formats are all supported
> - Use images that resemble the training domain for cleaner translations
> - Output is generated at 256×256 — upscale with other tools if needed

---

## Training Your Own Model (train.py)

You can train a new CycleGAN model on any two image domains — not just the three included here.

### Step 1 — Prepare your dataset

Create the following folder structure inside a `datasets/` directory:

```
datasets/
└── your_task_name/
    ├── trainA/     ← Domain A training images
    ├── trainB/     ← Domain B training images
    ├── testA/
    └── testB/
```

- Minimum ~100 images per domain (1000+ recommended)
- Images do **not** need to be paired or aligned

### Step 2 — Configure train.py

Open `train.py` and edit the configuration block at the top:

```python
DATASET = "your_task_name"     # Must match your folder name in datasets/
BATCH_SIZE = 1                 # 1–4 typical; reduce if GPU runs out of memory
LEARNING_RATE = 0.0002         # Default works well
NUM_EPOCHS = 200               # 200–400 recommended
DECAY_EPOCH = 100              # Epoch at which learning rate starts decaying
IMAGE_SIZE = 256               # Input resolution
LAMBDA_CYCLE = 10.0            # Cycle consistency loss weight
LAMBDA_IDENTITY = 0.5          # Identity loss weight
```

### Step 3 — Run training

```bash
python train.py
```

Training will automatically:

- Resume from the last saved checkpoint if one exists
- Save model checkpoints every 50 epochs to `checkpoints/your_task_name/`
- Save sample output images every 5 epochs to `samples/your_task_name/`

### Training output structure

```
checkpoints/your_task_name/
├── G_AB_epoch_050.pth     ← Generator: Domain A → Domain B
├── G_BA_epoch_050.pth     ← Generator: Domain B → Domain A
├── D_A_epoch_050.pth      ← Discriminator for Domain A
└── D_B_epoch_050.pth      ← Discriminator for Domain B

samples/your_task_name/epoch_050/
├── real_A.png
├── real_B.png
├── fake_B.png             ← A translated to B
└── fake_A.png             ← B translated to A
```

> **Training time estimates:**
>
> - GPU: ~2 min/epoch → ~6–8 hours for 200 epochs
> - CPU: ~20 min/epoch → very slow, GPU strongly recommended

---

## Datasets

Download the datasets from Kaggle and place them inside a `datasets/` folder in the project root:

| Model           | Dataset                   | Link                                                                      |
| --------------- | ------------------------- | ------------------------------------------------------------------------- |
| Horse ↔ Zebra   | `datasets/horse2zebra/`   | [Kaggle](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset)    |
| Monet ↔ Photo   | `datasets/monet2photo/`   | [Kaggle](https://www.kaggle.com/datasets/balraj98/monet2photo)            |
| Summer ↔ Winter | `datasets/summer2winter/` | [Kaggle](https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite) |

Each dataset should unzip into the structure:

```
datasets/horse2zebra/
├── trainA/
├── trainB/
├── testA/
└── testB/
```

> Datasets are not included in this repository due to file size. The pre-trained checkpoints are already included so you can run inference immediately without downloading any dataset.

---

## Project Structure

```
CycleGAN-Project/
├── README.md                    ← You are here
├── notes.txt                    ← Quick command reference
├── predict.py                   ← Run inference on images
├── train.py                     ← Train a new model
├── dataset.py                   ← Dataset loading logic
├── losses.py                    ← Loss function definitions
├── install_dependencies.py      ← Automated setup
├── models/
│   ├── generator.py             ← ResNet-based generator
│   └── discriminator.py         ← PatchGAN discriminator
├── checkpoints/                 ← Pre-trained model weights
│   ├── horse2zebra/
│   ├── summer2winter/
│   └── monet2photo/
├── samples/                     ← Training progress visualizations
├── results/                     ← 51+ pre-generated example outputs
└── datasets/                    ← Place downloaded datasets here
```

---

## Model Architecture

### Generator — ResNet-based (11.4M parameters)

```
Input (3 × 256 × 256)
  → Initial Conv (7×7, reflection padding)
  → Downsample ×2 (stride-2 convolutions)
  → 9 Residual Blocks
  → Upsample ×2 (ConvTranspose2d)
  → Output Conv (Tanh activation)
Output (3 × 256 × 256)
```

Uses Instance Normalization (not Batch Norm) for stable style transfer.

### Discriminator — PatchGAN (2.76M parameters)

```
Input (3 × 256 × 256)
  → 4 Conv blocks with LeakyReLU(0.2)
Output (1 × 30 × 30)  ← 900 patch-level real/fake predictions
```

Each output patch covers a 70×70 receptive field of the input.

### Loss Functions

| Loss              | Formula                    | Weight |
| ----------------- | -------------------------- | ------ |
| Adversarial       | Generator vs Discriminator | 1.0    |
| Cycle Consistency | `‖G_BA(G_AB(A)) − A‖₁`     | 10.0   |
| Identity          | `‖G_AB(B) − B‖₁`           | 0.5    |

---

## Example Outputs

51+ pre-generated results are in the `results/` directory:

- `h1–h9.png` — Horse ↔ Zebra translations
- `s1–s11.jpg`, `w1–w19.jpg` — Summer ↔ Winter examples
- `m1–m6.jpg`, `p1–p6.jpg` — Monet ↔ Photo examples
- `z4–z19.png`, `zeb1–zeb3.png` — Additional zebra results

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**

```bash
python install_dependencies.py
```

**`Image not found` error**
Use the full absolute path to your image:

```bash
--input /full/path/to/your/image.jpg
```

**`CUDA out of memory` during training**
Set `BATCH_SIZE = 1` in `train.py`, close other applications, or switch to CPU mode.

**Output image looks blurry**
The model outputs at 256×256 resolution by design. Use an upscaling tool like [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for higher resolution.

**Checkpoint file not found**
Double-check the exact path:

```bash
ls checkpoints/horse2zebra/
```

**Training not converging**
Verify your dataset folder structure matches the expected layout, try training for more epochs, or lower the learning rate slightly.

---

## References

- **Paper:** [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — Zhu et al., ICCV 2017
- **Official CycleGAN repo:** https://github.com/junyanz/CycleGAN
- **PyTorch:** https://pytorch.org
- GANs — Goodfellow et al. (2014)
- Instance Normalization — Ulyanov et al. (2016)
- PatchGAN — Isola et al. (2016)
