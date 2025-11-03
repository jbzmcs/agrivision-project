import os
import time
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------
# Model Selection
# ---------------------------
def select_model(num_classes):
    model = timm.create_model(
        'vit_base_patch16_siglip_224',  # Vision Transformer Base model
        pretrained=True,
        num_classes=num_classes
    )
    return model


# ---------------------------
# Configuration Setup
# ---------------------------
def config(model=None):
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss()
    if model is not None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return device, criterion, optimizer, scheduler
    else:
        return device, criterion


# ---------------------------
# Training Loop
# ---------------------------
def train_model(train_loader, val_loader, epochs, num_classes, base_dir='runs'):
    model = select_model(num_classes)
    device, criterion, optimizer, scheduler = config(model)
    model.to(device)

    print(f"\n Training on {device} using {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters\n")

    start_time = time.time()

    history = {
        'epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }
    save_dir = create_next_folder(base_dir, prefix='train_')

    for epoch in range(epochs):
        model, train_loss, train_acc = train(model, train_loader, device, optimizer, criterion)
        model, val_loss, val_acc = validate(model, val_loader, device, criterion)
        scheduler.step()

        elapsed_time = get_elapsed_time(start_time)
        history = update_history(history, epoch, epochs, train_loss, train_acc, val_loss, val_acc, model, elapsed_time, save_dir)
        plot_loss_accuracy(history, save_dir)

    print(f"\n Training complete. Best model and learning curves saved in: {save_dir}\n")
    return history


def train(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return model, running_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return model, running_loss / len(val_loader), 100 * correct / total


# ---------------------------
# Utilities
# ---------------------------
def update_history(history, epoch, epochs, train_loss, train_acc, val_loss, val_acc, model, elapsed_time, save_dir):
    history['epochs'].append(epoch + 1)
    history['train_losses'].append(train_loss)
    history['train_accuracies'].append(train_acc)
    history['val_losses'].append(val_loss)
    history['val_accuracies'].append(val_acc)

    print(f"Epoch [{epoch + 1}/{epochs}] @{elapsed_time} | "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save best model
    if len(history['val_losses']) == 1 or val_loss < min(history['val_losses'][:-1]):
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        print(f"💾 Saved best model at epoch {epoch + 1}")

    return history


def plot_loss_accuracy(history, save_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(history['epochs'], history['train_losses'], label='Train Loss', color='blue')
    ax[0].plot(history['epochs'], history['val_losses'], label='Val Loss', color='red')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history['epochs'], history['train_accuracies'], label='Train Acc', color='blue')
    ax[1].plot(history['epochs'], history['val_accuracies'], label='Val Acc', color='red')
    ax[1].set_title('Accuracy Curve')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()


def get_elapsed_time(start_time):
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    return f"{h:02}:{m:02}:{s:02}"


def create_next_folder(base_dir, prefix):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    runs = [int(f.replace(prefix, '')) for f in os.listdir(base_dir)
            if f.startswith(prefix) and f.replace(prefix, '').isdigit()]
    next_num = max(runs) + 1 if runs else 1
    save_path = os.path.join(base_dir, f"{prefix}{next_num}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


# ---------------------------
# Main Entry
# ---------------------------
if __name__ == "__main__":
    # Define image transforms with augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Define paths (robust path joining)
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    train_data = datasets.ImageFolder(os.path.join(base_path, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(base_path, "val"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(base_path, "test"), transform=transform)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

    # Auto-detect number of classes
    num_classes = len(train_data.classes)
    print(f"Detected {num_classes} classes: {train_data.classes}")

    # Train model
    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=200
    )
