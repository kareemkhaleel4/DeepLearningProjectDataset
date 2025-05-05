import os
import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from random import randint
import cv2

# Constants
GENERAL_CLASSES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(GENERAL_CLASSES)}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class FetalUltrasoundDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Image_name'] + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')
        label = CLASS_TO_IDX[self.df.loc[idx, 'Plane']]

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

# CNN model with adaptive pooling
class CNNImproved(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5):
        super(CNNImproved, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Data loading function
def load_data(csv_path, image_dir, batch_size=16):
    df = pd.read_csv(csv_path, delimiter=';')
    df = df[df['Plane'].isin(GENERAL_CLASSES)].copy()
    train_df = df[df['Train'] == 1]
    test_df = df[df['Train'] == 0]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = FetalUltrasoundDataset(train_df, image_dir, transform)
    test_dataset = FetalUltrasoundDataset(test_df, image_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_dataset

# Training function
def train_model(model, train_loader, test_loader, epochs=15, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_accuracies = [], []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, save_csv=True)
        train_losses.append(avg_loss)
        test_accuracies.append(acc)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

    return model

# Evaluation function with export
def evaluate(model, test_loader, save_csv=False):
    model.eval()
    correct, total = 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    report = classification_report(targets, preds, target_names=GENERAL_CLASSES, output_dict=True)
    cm = confusion_matrix(targets, preds)

    if save_csv:
        pd.DataFrame(report).transpose().to_csv("classification_report.csv")
        pd.DataFrame(cm, index=GENERAL_CLASSES, columns=GENERAL_CLASSES).to_csv("confusion_matrix.csv")

    return accuracy

# Visualization and SHAP functions
def show_transformed_images(dataset, num_images=5):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        x = randint(0, len(dataset))
        img, label, _ = dataset[x]
        axs[i].imshow(img.squeeze(0), cmap='gray')
        axs[i].set_title(GENERAL_CLASSES[label])
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig("transformed_samples.png")
    plt.show()

def get_misclassified(model, dataset):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label, name = dataset[i]
            input_tensor = img.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            if pred != label:
                misclassified.append((img, label, pred, name))
    return misclassified

def show_misclassified_images(misclassified, num_images=5):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(min(num_images, len(misclassified))):
        img, true_label, pred_label, name = misclassified[i]
        axs[i].imshow(img.squeeze(0), cmap='gray')
        axs[i].set_title(f'True: {GENERAL_CLASSES[true_label]}\nPred: {GENERAL_CLASSES[pred_label]}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig("misclassified_samples.png")
    plt.show()

def explain_with_shap(model, dataset, num_samples=3):
    
    model.eval()
    background = torch.cat([dataset[i][0].unsqueeze(0) for i in range(10)], dim=0).to(device)
    explainer = shap.GradientExplainer(model, background)
    test_images = torch.cat([dataset[i][0].unsqueeze(0) for i in range(num_samples)], dim=0).to(device)
    shap_values = explainer.shap_values(test_images)

    for i in range(num_samples):
        img_np = test_images[i].cpu().numpy().transpose(1, 2, 0)
        if isinstance(shap_values, list) and isinstance(shap_values[0], np.ndarray):
            if shap_values[0].ndim == 4:
                shap_map = shap_values[0][i].mean(axis=0)
            else:
                shap_map = np.stack([sv[i] for sv in shap_values], axis=0).mean(axis=0)

            # Normalize and resize for overlay
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
            shap_map_resized = cv2.resize(shap_map, (img_np.shape[1], img_np.shape[0]))
            overlay = (shap_map_resized * 255).astype(np.uint8)
            overlay_colored = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
            img_gray = (img_np.squeeze() * 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            combined = cv2.addWeighted(img_gray, 0.6, overlay_colored, 0.4, 0)

            plt.figure(figsize=(4, 4))
            plt.imshow(combined)
            plt.title("SHAP Highlight")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("Unexpected SHAP values format.")


# Main
if __name__ == '__main__':
    csv_path = 'FETAL_PLANES_DB_data.csv'
    image_dir = './Images/'
    batch_size = 32
    train_loader, test_loader, test_dataset = load_data(csv_path, image_dir, batch_size)

    model = CNNImproved().to(device)
    model = train_model(model, train_loader, test_loader, epochs=5, lr=1e-4)

    print("\n✅ Showing transformed images:")
    show_transformed_images(test_dataset, num_images=5)

    print("\n❌ Showing misclassified examples:")
    misclassified = get_misclassified(model, test_dataset)
    show_misclassified_images(misclassified, num_images=5)

    print("\n🔍 SHAP Explainability:")
    explain_with_shap(model, test_dataset, num_samples=3)
