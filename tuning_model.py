import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
GENERAL_CLASSES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur',
                   'Fetal thorax', 'Maternal cervix', 'Other']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(GENERAL_CLASSES)}

csv_path = 'FETAL_PLANES_DB_data.csv'
image_dir = 'Images/'
save_results_csv = 'res_data/hyperparameter_results.csv'
save_cm_dir = 'res_data/confusion_matrices/'

# Ensure save directory exists
os.makedirs(save_cm_dir, exist_ok=True)

# Dataset
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

        return image, label

# Load data
def load_data(csv_path, image_dir, batch_size, transform):
    df = pd.read_csv(csv_path, delimiter=';')
    df = df[df['Plane'].isin(GENERAL_CLASSES)].copy()

    train_df = df[df['Train'] == 1]
    test_df = df[df['Train'] == 0]

    train_dataset = FetalUltrasoundDataset(train_df, image_dir, transform)
    test_dataset = FetalUltrasoundDataset(test_df, image_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_df

# Model
class CNNBaseline(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Evaluation
def evaluate(model, loader, config_name):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    cm = confusion_matrix(labels_all, preds)
    report = classification_report(labels_all, preds, target_names=GENERAL_CLASSES, output_dict=True)
    print(cm)
    print( report)
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GENERAL_CLASSES, yticklabels=GENERAL_CLASSES)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix: {config_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_cm_dir, f'{config_name}.png'))
    plt.close()

    return report['accuracy'], report

# Training
def train_and_evaluate(config):
    lr, dropout, batch_size = config
    config_name = f"lr_{lr}_dropout_{dropout}_bs_{batch_size}"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_loader, test_loader, _ = load_data(csv_path, image_dir, batch_size, transform)

    model = CNNBaseline(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Short training (5 epochs) for tuning
    for epoch in range(10):
        model.train()
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{10}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    accuracy, report = evaluate(model, test_loader, config_name)
    print(f'accuracy: {accuracy:.4f}')
    return {
        'config': config_name,
        'accuracy': accuracy,
        'macro avg f1-score': report['macro avg']['f1-score'],
        'weighted avg f1-score': report['weighted avg']['f1-score']
    }

# Grid Search Configs
learning_rates = [1e-3, 1e-4]
dropouts = [0.3, 0.5]
batch_sizes = [16, 32]

results = []
for config in itertools.product(learning_rates, dropouts, batch_sizes):
    print(f"Training with config: {config}")
    result = train_and_evaluate(config)
    results.append(result)

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(save_results_csv, index=False)
df_results
