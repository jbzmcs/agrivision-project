import torch
from train_plant_vit import config, create_next_folder, select_model
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def load_dataset(input_size=(224,224), batch_size=64,
                 train_dir='./data/train',
                 val_dir='./data/val',
                 test_dir='./data/test'):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f" Loaded test dataset: {len(test_data)} images, {len(test_data.classes)} classes")
    return test_loader



def test_model(test_loader, model_path, num_classes, k=5):
   
    device, criterion = config()
    model = select_model(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    evaluate(model, test_loader, device, criterion, num_classes, k)


def evaluate(model, data_loader, device, criterion, num_classes, k, base_dir='runs'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = torch.tensor([], dtype=torch.long).to(device)
    all_preds = torch.tensor([], dtype=torch.long).to(device)
    all_probs = torch.tensor([], dtype=torch.float).to(device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, predicted))
            all_probs = torch.cat((all_probs, torch.softmax(outputs, dim=1)))

    loss = running_loss / len(data_loader)
    acc = 100 * correct / total
    all_labels_cpu = all_labels.cpu().numpy()
    all_preds_cpu = all_preds.cpu().numpy()
    all_probs_cpu = all_probs.cpu().numpy()

    save_dir = create_next_folder(base_dir, prefix='test_')
    show_loss_accuracy(loss, acc, num_classes, all_labels_cpu, all_probs_cpu, all_preds_cpu, k, save_dir)
    get_classification_report(all_labels_cpu, all_preds_cpu, save_dir)
    get_confusion_matrix(all_labels_cpu, all_preds_cpu, save_dir)
    get_roc_auc_score(num_classes, all_labels_cpu, all_probs_cpu, save_dir)
    get_precision_recall_curve(num_classes, all_labels_cpu, all_probs_cpu, save_dir)

    print(f" Evaluation complete. Results saved in: {save_dir}")
    return model



def show_loss_accuracy(loss, acc, num_classes, all_labels, all_probs, all_preds, k, save_dir):
    top_k_acc = metrics.top_k_accuracy_score(all_labels, all_probs, k=k, labels=range(num_classes))
    mcc = metrics.matthews_corrcoef(all_labels, all_preds)
    bal_acc = metrics.balanced_accuracy_score(all_labels, all_preds)

    metrics_data = {
        'Loss': [loss],
        'Accuracy': [acc],
        f'Top-{k} Accuracy': [top_k_acc],
        'Balanced Accuracy': [bal_acc],
        'MCC': [mcc]
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = [[m, v[0]] for m, v in metrics_data.items()]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title('Model Evaluation Metrics', fontsize=14)
    plt.savefig(os.path.join(save_dir, 'Model_Evaluation_Metrics.png'))
    plt.close()


def get_classification_report(all_labels, all_preds, save_dir):
    report = metrics.classification_report(all_labels, all_preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt='.3f')
    plt.title('Classification Report')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.savefig(os.path.join(save_dir, 'Classification_Report.png'))
    plt.close()


def get_confusion_matrix(all_labels, all_preds, save_dir):
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'Confusion_Matrix.png'))
    plt.close()


def get_roc_auc_score(num_classes, all_labels, all_probs, save_dir):
    y_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'ROC_Curve.png'))
    plt.close()


def get_precision_recall_curve(num_classes, all_labels, all_probs, save_dir):
    all_labels_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    plt.figure()
    for i in range(num_classes):
        precision, recall, _ = metrics.precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
        auc_score = metrics.auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_dir, 'Precision_Recall_Curve.png'))
    plt.close()


\

if __name__ == "__main__":
    test_loader = load_dataset(
        input_size=(224,224),
        batch_size=32,
        test_dir = "../data/test"

    )

    test_model(
        test_loader,
        model_path = "./runs/train_9/best_model.pt",
        num_classes=10,
    )
