import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model import UNet
from dataset import get_loaders
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = F.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return num_correct/num_pixels, dice_score/len(loader)

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    # Hyperparameters
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    NUM_WORKERS = 4
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    PIN_MEMORY = True
    LOAD_MODEL = False
    TRAIN_IMG_DIR = "data/train/images"
    TRAIN_MASK_DIR = "data/train/masks"
    VAL_IMG_DIR = "data/val/images"
    VAL_MASK_DIR = "data/val/masks"

    # Create directories if they don't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Get data loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Initialize model, loss function, optimizer
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter("runs/skin_tissue_segmentation")

    # Load checkpoint if needed
    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"checkpoints/checkpoint_epoch_{epoch+1}.pth.tar")

        # Check accuracy
        print("Checking accuracy on validation set...")
        val_acc, val_dice = check_accuracy(val_loader, model, device=DEVICE)
        
        # Log metrics
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)
        writer.add_scalar("Validation/Dice_Score", val_dice, epoch)

    writer.close()

if __name__ == "__main__":
    main()