import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy

# --- Configuration & Hyperparameters ---
DATA_DIR = 'data'
# --- CHANGED: New output paths ---
MODEL_SAVE_PATH = 'models/alexnet_overfit.pth' 
OUTPUT_DIR = 'output/alexnet_overfit'
# -----------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 200 # Max epochs (it will run all of them)
LEARNING_RATE = 0.001
# --- REMOVED: PATIENCE constant ---
# ----------------------------------

def print_config():
    """
    Prints the script's configuration to the console.
    """
    print("\n--- Configuration (Overfit Experiment) ---")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Model Save Path: {MODEL_SAVE_PATH}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("  EARLY STOPPING: DISABLED")
    print("------------------------------------------\n")

def setup_device():
    # (This function is identical to our previous script)
    if not torch.cuda.is_available():
        print("\n--- ERROR ---")
        print("CUDA is not available. This script requires a GPU to run.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    return device

def get_data_loaders(data_dir, batch_size):
    # (This function is identical to our previous script)
    print(f"Loading data from '{data_dir}'...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    return dataloaders, num_classes

def build_model(num_classes):
    # (This function is identical to our previous script)
    print("Loading pre-trained AlexNet model...")
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model

def train_one_epoch(model, loader, device, criterion, optimizer):
    # (This function is identical to our previous script)
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()

def validate_one_epoch(model, loader, device, criterion):
    # (This function is identical to our previous script)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()

def save_learning_curve(history, output_path):
    # (This function is identical to our previous script)
    print(f"Saving learning curve to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-o', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_training_loop(model, criterion, optimizer, dataloaders, device, num_epochs):
    """
    Executes the main training loop.
    --- MODIFIED: This version runs for ALL epochs and returns the FINAL model. ---
    """
    print("\nStarting training (Full 200 Epochs)...")
    start_time = time.time()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0 # We still track this, just for the final print message

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], device, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, dataloaders['val'], device, criterion)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # We're still tracking the best accuracy, just not stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy (for comparison): {best_val_acc:.4f}')
    print(f'Final validation accuracy (at epoch 200): {val_acc:.4f}')
    
    # --- CHANGED: Return the FINAL model state, not the best one ---
    return model.state_dict(), history

def save_model(model_weights, save_path):
    # (This function is identical to our previous script)
    print(f"\nSaving final model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_weights, save_path)
    print("Model saved successfully.")

def print_model_summary(model):
    """
    Calculates and prints the total and trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- Model Summary ---")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-Trainable Parameters: {total_params - trainable_params:,}")
    print("-----------------------\n")    

def main():
    """
    Orchestrates the training and validation process.
    """
    print(f"Starting AlexNet OVERFIT Experiment")
    
    # 1. Setup
    print_config()
    device = setup_device()
    dataloaders, num_classes = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # 2. Build Model
    model = build_model(num_classes)
    model = model.to(device)

    print_model_summary(model)

    # 3. Define Optimizer and Loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Run Training
    # --- CHANGED: 'patience' argument is removed ---
    final_model_wts, history = run_training_loop(
        model, criterion, optimizer, dataloaders, 
        device, NUM_EPOCHS
    )

    # 5. Save Artifacts
    # --- CHANGED: Saves the FINAL model weights ---
    save_model(final_model_wts, MODEL_SAVE_PATH)
    
    plot_path = os.path.join(OUTPUT_DIR, 'learning_curve.png')
    save_learning_curve(history, plot_path)
    
    print("\nAlexNet OVERFIT experiment complete.")

if __name__ == '__main__':
    main()