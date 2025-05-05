# Baseline Fetal CNN - Python Script Version

import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create result directory
os.makedirs("res_data", exist_ok=True)

# Dataset preparation function
def prepare_fetal_planes_dataset(csv_path, images_folder, output_folder='./data/split', test_size=0.3, random_seed=2025):
    df = pd.read_csv(csv_path, sep=';')

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, stratify=df['Plane'])

    def copy_files(df, split_folder):
        for _, row in df.iterrows():
            label = row['Plane'].replace(' ', '_')
            src = os.path.join(images_folder, row['Image_name'] + '.png')
            dst_dir = os.path.join(output_folder, split_folder, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, row['Image_name'] + '.png')
            shutil.copy(src, dst)

    copy_files(train_df, 'train')
    copy_files(test_df, 'test')
    print("Dataset preparation complete.")

# Evaluation metrics function
def evaluate_model(model, dataloader, class_names):
    model.eval()
    y_true, y_pred = [], []
    misclassified_imgs = []
    misclassified_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            for i in range(len(labels)):
                if labels[i] != predicted[i]:
                    misclassified_imgs.append(inputs[i].cpu())
                    misclassified_labels.append((labels[i].item(), predicted[i].item()))

    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv("res_data/classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("res_data/confusion_matrix.png")
    plt.close()

    if misclassified_imgs:
        fig, axes = plt.subplots(1, min(6, len(misclassified_imgs)), figsize=(15, 5))
        for idx, ax in enumerate(axes):
            img = misclassified_imgs[idx].permute(1, 2, 0).numpy()
            true_label, pred_label = misclassified_labels[idx]
            ax.imshow(img)
            ax.set_title(f"T: {class_names[true_label]}\nP: {class_names[pred_label]}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("res_data/misclassified_samples.png")
        plt.close()

# CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Main script execution
def main():
    # Transforms
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_dataset_full = datasets.ImageFolder('./data/train', transform=transform)
    test_dataset = datasets.ImageFolder('./data/test', transform=transform)
    class_names = train_dataset_full.classes

    # Split train/val
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model setup
    model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train loop
    print("start")
    start_time = time.time()
    for epoch in range(1, 11):
        epoch_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch} | Train Loss: {running_loss/total:.4f} | Accuracy: {correct/total:.4f} | time for this epoch {time.time()-epoch_time:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        print(f"Validation Loss: {val_loss/val_total:.4f} | Accuracy: {val_correct/val_total:.4f}")
    # Final evaluation
    print("\nEvaluating on test set:")
    evaluate_model(model, test_loader, class_names)
    final_time = time.time()-start_time
    print(f"Full time = {final_time:.4f}")
    torch.save(model.state_dict(), 'res_data/simple_cnn_fetal_planes.pth')

if __name__ == '__main__':
    main()
