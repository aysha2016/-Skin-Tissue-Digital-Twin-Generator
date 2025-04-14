# Skin-Tissue Digital Twin Generator

## Overview
This tool reconstructs internal skin layers (epidermis, dermis, blood vessels) from surface optical images using a U-Net model trained on annotated histology datasets.

## How to Use
1. Place training images in `data/train/images` and masks in `data/train/masks`.
2. Run `train.py` to train the U-Net.
3. Use `predict.py` to run inference on a new image.

## Requirements
- PyTorch
- torchvision
- Pillow