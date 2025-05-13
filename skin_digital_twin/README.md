# Skin-Tissue Digital Twin Generator

## Overview
This tool reconstructs internal skin layers—epidermis, dermis, and blood vessels—from surface-level optical skin images using a U-Net model trained on annotated histology data.

Ideal for dermatological simulation, medical diagnostics, and digital twin applications.

## Features
- U-Net architecture for pixel-wise tissue segmentation
- Multi-class label reconstruction: epidermis, dermis, blood vessels
- Trained on paired surface and histological cross-section datasets
- Image preprocessing and postprocessing pipeline

## How to Use
1. Place training images in `data/train/images` and masks in `data/train/masks`.
2. Run `train.py` to train the U-Net.
3. Use `predict.py` to run inference on a new image.

## Requirements
- PyTorch
- torchvision
- Pillow