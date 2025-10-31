import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import time
import os

# --- Configuration & Hyperparameters ---
# Project Objective: These MUST be identical to train_alexnet.py
DATA_DIR = 'data'
MODEL_SAVE_PATH = 'models/resnet50_final.pth' # <-- CHANGED
NUM_CLASSES = 4       
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
# ---------------------------------------

def main():
    """
    Main function to execute the ResNet50 training and evaluation pipeline.
    Fulfills Objective 3.
    """
    print(f"Starting Objective 3: ResNet50 Training")
    print(f"Configuration: {NUM_CLASSES=}, {BATCH_SIZE=}, {NUM_EPOCHS=}, {LEARNING_RATE=}")

    # 1. Setup Device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Data Transforms (Identical to AlexNet)
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
        ]),
    }

    # 3. Load Data using ImageFolder (Identical to AlexNet)
    print(f"Loading data from '{DATA_DIR}'...")
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # 4. Load Pre-trained ResNet50 and Modify for Transfer Learning
    # --- THIS SECTION IS THE ONLY MAJOR CHANGE ---
    print("Loading pre-trained ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # <-- CHANGED

    # Freeze all parameters in the model (convolutional base)
    for param in model.parameters(): # <-- CHANGED
        param.requires_grad = False

    # Get the number of input features for the final fully connected layer
    num_ftrs = model.fc.in_features # <-- CHANGED

    # Replace the final layer (model.fc) with a new, unfrozen
    # Linear layer that matches our number of classes.
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # <-- CHANGED
    # --- END OF MAJOR CHANGE SECTION ---

    # Move the model to the configured device (GPU/CPU)
    model = model.to(device)

    # 5. Define Loss Function and Optimizer (Identical Logic)
    # We filter for only parameters where requires_grad == True
    # In this case, it will only be the new `model.fc` parameters
    params_to_update = model.parameters()
    print("Parameters to update:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t{name}")
            
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, params_to_update), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 6. Training Loop (Identical to AlexNet)
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 7. Final Evaluation (Identical Logic, Objective 4 Requirement)
    print("\n--- Final Model Evaluation ---")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report (ResNet50):") 
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    report_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'resnet50_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    print("\nConfusion Matrix (ResNet50):") 
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("\nGenerating confusion matrix plot...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Resnet50 Confusion Matrix")
    plt.show()  # This will pause the script and open the plot window

    # 8. Save the Trained Model (Identical Logic)
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("ResNet50 training and evaluation complete.")

if __name__ == '__main__':
    main()