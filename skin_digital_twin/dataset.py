import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinTissueDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Define default transforms
        if self.transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.Resize(256, 256),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(256, 256),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # Assuming grayscale masks
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            
            # Convert mask to long tensor for CrossEntropyLoss
            mask = mask.long()

        return image, mask

def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform=None,
    val_transform=None,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SkinTissueDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        is_train=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SkinTissueDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        is_train=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader