import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np
import os
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     roc_curve,
#     auc,
#     precision_recall_curve,
#     average_precision_score
# )


# --- Configuration ---
DATA_DIR = 'data'
MODEL_PATH = 'models/alexnet_best.pth' # Path to the *best saved* model
OUTPUT_DIR = 'output/alexnet'          # Output directory for reports
BATCH_SIZE = 64
# ---------------------

def setup_device():
    """
    Checks for CUDA, prints device info, and returns the device.
    """
    if not torch.cuda.is_available():
        print("\n--- ERROR ---")
        print("CUDA is not available. This script requires a GPU to run.")
        print("Please check your PyTorch installation and NVIDIA drivers.")
        sys.exit(1)
        
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    return device

def get_test_loader(data_dir, batch_size):
    """
    Loads the test dataset and returns the loader and class names.
    """
    print(f"Loading data from '{data_dir}'...")
    
    # --- Standard ImageNet normalization ---
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # We load 'train' *only* to get class names for label consistency
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    except FileNotFoundError:
        print(f"Error: 'data/train' directory not found.")
        sys.exit(1)
        
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Load the actual test dataset
    try:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms)
    except FileNotFoundError:
        print(f"Error: 'data/test' directory not found.")
        sys.exit(1)
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Test dataset loaded: {len(test_dataset)} images.")
    
    return test_loader, class_names, num_classes

def build_model(model_path, num_classes):
    """
    Initializes the model structure and loads the saved weights.
    """
    print("Loading model structure...")
    model = models.alexnet(weights=None) # Start with an untrained model
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    # Load Trained Weights
    try:
        # Use weights_only=True for security
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_alexnet.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model state: {e}")
        sys.exit(1)
        
    model.eval() # Set model to evaluation mode
    return model

def run_evaluation(model, loader, device, criterion, top_k=5):
    """
    Runs evaluation on the test set and returns all labels,
    predictions, probabilities, average loss, and top-1 accuracy.
    """
    print("Running evaluation on test set...")
    all_labels, all_preds_top1, all_probs = [], [], []
    running_loss, correct_top1, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels) # Calculate loss
            probs = torch.softmax(outputs, dim=1) # Get probabilities
            _, preds_top1_batch = torch.max(outputs, 1)

            # Accumulate results
            all_preds_top1.extend(preds_top1_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Accumulate loss and metrics
            running_loss += loss.item() * inputs.size(0)
            correct_top1 += (preds_top1_batch == labels).sum().item()
            total += labels.size(0)

    # Calculate final metrics
    avg_loss = running_loss / total
    acc_top1 = correct_top1 / total # This is the raw float (0.0-1.0)
    
    print(f"\n--- Model Evaluation Complete ---")
    print(f"  Average Test Loss: {avg_loss:.4f}")
    print(f"  Top-1 Accuracy: {acc_top1*100:.2f}%")
    
    return np.array(all_labels), np.array(all_preds_top1), np.array(all_probs), avg_loss, acc_top1

def save_evaluation_metrics_table(avg_loss, acc_top1, num_classes, labels, probs, preds, k, output_dir, model_name="AlexNet"):
    """
    Calculates key evaluation metrics and saves them as a clean PNG table,
    matching the teammate's visual style.
    """
    print(f"\nGenerating and saving {model_name} evaluation metrics table...")
    
    # 1. Calculate remaining metrics
    acc_topk = metrics.top_k_accuracy_score(labels, probs, k=k, labels=np.arange(num_classes))
    mcc = metrics.matthews_corrcoef(labels, preds)
    bal_acc = metrics.balanced_accuracy_score(labels, preds)

    # 2. Format metrics for the table (matching teammate's unformatted style)
    metrics_data = {
        'Loss': [avg_loss],
        'Accuracy': [acc_top1], # This is Top-1
        f'Top-{k} Accuracy': [acc_topk],
        'Balanced Accuracy': [bal_acc],
        'MCC': [mcc]
    }

    # 3. Plot the table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = [[m, v[0]] for m, v in metrics_data.items()]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    plt.title(f'{model_name} Model Evaluation Metrics', fontsize=16)
    
    # 4. Save the plot
    save_path = os.path.join(output_dir, 'model_evaluation_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Metrics table saved to: {save_path}")

def plot_classification_report_heatmap(labels, preds, probs, class_names, num_classes, output_dir, model_name="AlexNet", k=5):
    """
    Generates and saves a visual heatmap of the classification report.
    It now calculates its own Top-k metrics for the title.
    """
    print(f"\nGenerating and saving {model_name} classification report heatmap...")
    
    # 1. Calculate metrics needed for title
    # --- FIX: Added 'metrics.' prefix ---
    acc_top1 = metrics.accuracy_score(labels, preds)
    acc_topk = metrics.top_k_accuracy_score(labels, probs, k=k, labels=np.arange(num_classes))
    # ------------------------------------
    
    # 2. Generate the standard text report
    # --- FIX: Added 'metrics.' prefix ---
    report = metrics.classification_report(labels, preds, target_names=class_names, digits=4, output_dict=True)
    # ------------------------------------
    df_report = pd.DataFrame(report).transpose()
    
    # Get class names from the report, excluding summary rows
    report_class_names = [name for name in df_report.index.values if name not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_to_plot = df_report.loc[report_class_names, ['precision', 'recall', 'f1-score']]
    
    # 3. Create and save the heatmap
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(
        metrics_to_plot,
        annot=True,
        cmap='Blues',
        fmt=".3f", 
        linewidths=.5,
        cbar_kws={'label': 'Score'}
    )
    plt.title(f'{model_name} Classification Report Heatmap (Test Set)\n'
              f'Top-1 Accuracy: {acc_top1*100:.2f}% | Top-5 Accuracy: {acc_topk*100:.2f}%', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.yticks(rotation=0) 
    plt.tight_layout()
    
    heatmap_save_path = os.path.join(output_dir, 'classification_report_heatmap.png')
    plt.savefig(heatmap_save_path, dpi=300)
    plt.close()
    print(f"Classification report heatmap saved to: {heatmap_save_path}")

    # 4. Also save the traditional text report
    full_report_text = f"{model_name} Classification Report (Test Set)\n"
    full_report_text += "="*40 + "\n"
    full_report_text += f"  Top-1 Accuracy: {acc_top1*100:.2f}%\n"
    full_report_text += f"  Top-5 Accuracy: {acc_topk*100:.2f}%\n"
    full_report_text += "="*40 + "\n\n"
    
    # --- FIX: Added 'metrics.' prefix ---
    full_report_text += metrics.classification_report(labels, preds, target_names=class_names, digits=4)
    # ------------------------------------
    
    text_report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(text_report_path, 'w') as f:
        f.write(full_report_text)
    print(f"Traditional text report saved to: {text_report_path}")
    print("\n--- Console Classification Report ---")
    print(full_report_text)


def save_confusion_matrix(labels, preds, class_names, output_dir):
    """
    Generates and saves the confusion matrix plot.
    """
    print("\nGenerating and saving confusion matrix plot...")
    
    # --- FIX: Added 'metrics.' prefix ---
    cm = metrics.confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # --- FIX: Added 'metrics.' prefix ---
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # -----------------------------------
    
    display.plot(ax=ax, xticks_rotation=45, cmap='viridis')
    
    plt.title("AlexNet Confusion Matrix (Test Set)", fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    plt.tight_layout()

    cm_save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_save_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_save_path}")


def save_roc_curves(labels, probs, class_names, num_classes, output_dir):
    """
    Generates and saves the one-vs-rest (OvR) ROC curves.
    """
    print("\nGenerating and saving ROC curves...")
    plt.figure(figsize=(12, 10))
    
    for i in range(num_classes):
        y_true_bin = (labels == i).astype(int)
        y_score = probs[:, i]
        
        # --- FIX: Added 'metrics.' prefix ---
        fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        # -----------------------------------
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_save_path, dpi=300)
    plt.close()
    print(f"ROC curves saved to {roc_save_path}")

def save_pr_curves(labels, probs, class_names, num_classes, output_dir):
    """
    Generates and saves the one-vs-rest (OvR) Precision-Recall curves.
    """
    print("\nGenerating and saving Precision-Recall curves...")
    plt.figure(figsize=(12, 10))
    
    for i in range(num_classes):
        y_true_bin = (labels == i).astype(int)
        y_score = probs[:, i]
        
        # --- FIX: Added 'metrics.' prefix ---
        precision, recall, _ = metrics.precision_recall_curve(y_true_bin, y_score)
        avg_precision = metrics.average_precision_score(y_true_bin, y_score)
        # -----------------------------------
        
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Test Set', fontsize=16)
    plt.legend(loc="lower left")
    plt.grid(True)
    pr_save_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_save_path, dpi=300)
    plt.close()
    print(f"Precision-Recall curves saved to {pr_save_path}")

def main():
    """
    Orchestrates the testing and evaluation process.
    """
    print(f"Starting AlexNet Evaluation (Refactored)")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output dir exists
    
    # 1. Setup
    device = setup_device()
    test_loader, class_names, num_classes = get_test_loader(DATA_DIR, BATCH_SIZE)
    
    criterion = nn.CrossEntropyLoss()
    # 2. Build Model
    model = build_model(MODEL_PATH, num_classes)
    model = model.to(device)

    # 3. Run Evaluation
    labels, preds_top1, probs, avg_loss, acc_top1 = run_evaluation(
        model, test_loader, device, criterion, top_k=5 
    )

    # 4. Save Artifacts
    save_evaluation_metrics_table(
        avg_loss, acc_top1, num_classes, labels, 
        probs, preds_top1, k=5, output_dir=OUTPUT_DIR, model_name="AlexNet"
    )

    # Save the heatmap report
    plot_classification_report_heatmap(
        labels, preds_top1, probs, class_names, 
        num_classes, OUTPUT_DIR, model_name="AlexNet", k=5
    )
    
    # Save the rest of the plots
    save_confusion_matrix(labels, preds_top1, class_names, OUTPUT_DIR)
    save_roc_curves(labels, probs, class_names, num_classes, OUTPUT_DIR)
    save_pr_curves(labels, probs, class_names, num_classes, OUTPUT_DIR)
    
    print("\nAlexNet evaluation complete.")

if __name__ == '__main__':
    main()