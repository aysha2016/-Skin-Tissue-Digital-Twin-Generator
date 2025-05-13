import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Load and preprocess image for prediction"""
    image = np.array(Image.open(image_path).convert("RGB"))
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0), image

def predict_image(model, image_path, device="cuda", save_path=None):
    """Run prediction on a single image"""
    model.eval()
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions = model(image_tensor)
        predictions = F.softmax(predictions, dim=1)
        pred_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()

    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Prediction mask
    plt.subplot(132)
    plt.imshow(pred_mask, cmap="viridis")
    plt.title("Predicted Segmentation")
    plt.axis("off")

    # Overlay
    plt.subplot(133)
    overlay = original_image.copy()
    overlay[pred_mask == 1] = [255, 0, 0]  # Red for epidermis
    overlay[pred_mask == 2] = [0, 255, 0]  # Green for dermis
    overlay[pred_mask == 3] = [0, 0, 255]  # Blue for blood vessels
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "checkpoints/my_checkpoint.pth.tar"
    IMAGE_PATH = "data/test/images/test_image.jpg"  # Update with your test image path
    SAVE_PATH = "predictions/prediction.png"

    # Load model
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Run prediction
    predict_image(
        model=model,
        image_path=IMAGE_PATH,
        device=DEVICE,
        save_path=SAVE_PATH
    )

if __name__ == "__main__":
    main()