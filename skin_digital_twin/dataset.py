import os
from PIL import Image
from torch.utils.data import Dataset

class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.masks = sorted(os.listdir(os.path.join(root_dir, "masks")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask