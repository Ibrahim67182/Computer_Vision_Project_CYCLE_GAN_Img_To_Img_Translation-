# predict.py — load a trained CycleGAN generator and translate any input image

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image, ImageEnhance
import os
import argparse

from models import generator


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="CycleGAN Inference")

parser.add_argument("--input",      type=str, required=True,  help="Path to input image")
parser.add_argument("--checkpoint", type=str, required=True,  help="Path to generator .pth file")
parser.add_argument("--output",     type=str, default="output.png", help="Path to save output image")
parser.add_argument("--mode", type=str, default="horse2zebra", choices=["horse2zebra", "summer2winter","monet2photo"], help="Translation mode")

args = parser.parse_args()


# ══════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ══════════════════════════════════════════════════════════════
# LOAD GENERATOR
# ══════════════════════════════════════════════════════════════

G = generator.Generator().to(device)
G.load_state_dict(torch.load(args.checkpoint, map_location=device))
G.eval()

print(f"Checkpoint loaded: {args.checkpoint}")


# ══════════════════════════════════════════════════════════════
# IMAGE TRANSFORMS
# ══════════════════════════════════════════════════════════════

transform = transforms.Compose([
    transforms.Resize((256, 256)),          # resize to model input size
    transforms.ToTensor(),                  # [0,255] → [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],  # [0,1]  → [-1,1]
                         [0.5, 0.5, 0.5])
])


# ══════════════════════════════════════════════════════════════
# LOAD AND TRANSLATE IMAGE
# ══════════════════════════════════════════════════════════════

# Load input image
input_image = Image.open(args.input).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)  # [1, 3, 256, 256]

print(f"Input image loaded: {args.input}")

# Run through generator
with torch.no_grad():
    output_tensor = G(input_tensor)         # [-1, 1]

# Denormalize: [-1,1] → [0,1]
output_tensor = output_tensor * 0.5 + 0.5

# Save output
os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
save_image(output_tensor, args.output)



# ══════════════════════════════════════════════════════════════
# POST-PROCESSING ENHANCEMENT
# ══════════════════════════════════════════════════════════════

img = Image.open(args.output)

if args.mode == "horse2zebra":
    img = ImageEnhance.Sharpness(img).enhance(1.8)   # fix blurriness
    img = ImageEnhance.Color(img).enhance(1.5)        # fix washed out colors
    img = ImageEnhance.Contrast(img).enhance(1.2)     # punch up contrast
    img = ImageEnhance.Brightness(img).enhance(1.05)  # slight brightness lift

elif args.mode == "summer2winter":
    img = ImageEnhance.Sharpness(img).enhance(2.0)   # fix blurriness
    img = ImageEnhance.Color(img).enhance(1.7)        # fix washed out colors
    img = ImageEnhance.Contrast(img).enhance(1.0)     # punch up contrast
    img = ImageEnhance.Brightness(img).enhance(1.0)  # slight brightness lift

elif args.mode == "monet2photo":
    img = ImageEnhance.Sharpness(img).enhance(1.9)    # aggressive sharpening for painterly blur
    img = ImageEnhance.Color(img).enhance(1.8)         # strong color boost — fix washed-out palette
    img = ImageEnhance.Contrast(img).enhance(1.3)      # lift contrast to separate tones
    img = ImageEnhance.Brightness(img).enhance(0.95)   # slight pull-back — prevents blown highlights


img.save(args.output)

print(f"Output saved: {args.output}")