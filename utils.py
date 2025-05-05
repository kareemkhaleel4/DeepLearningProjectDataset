#Add all imports required
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



# Define your classes
GENERAL_CLASSES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 
                   'Fetal thorax', 'Maternal cervix', 'Other']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(GENERAL_CLASSES)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
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

# Data loading and splitting
def load_data(csv_path, image_dir, batch_size=32, transform=None):
    df = pd.read_csv(csv_path, delimiter=';')
    df = df[df['Plane'].isin(GENERAL_CLASSES)].copy()

    train_df = df[df['Train'] == 1]
    test_df = df[df['Train'] == 0]

    if not transform:
        transform = transforms.Compose([#transformer 1
            transforms.Resize((128, 128)), #// original
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    train_dataset = FetalUltrasoundDataset(train_df, image_dir, transform)
    test_dataset = FetalUltrasoundDataset(test_df, image_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



# CNN Model Definition
class CNNBaseline(nn.Module):
    def __init__(self, num_classes=6):
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
#            nn.Linear(256 * 8 * 8, 512), # transformer 1
            nn.Linear(256 * 14 * 14, 512), # transformer 2 and 3
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNNImproved(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5):
        super(CNNImproved, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Metrics plotting
def plot_metrics(losses, accuracies):
    epochs = range(1, len(losses)+1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, losses, '-o', label='Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss'); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracies, '-o', label='Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Test Accuracy'); plt.grid(True)

    plt.tight_layout()
    plt.show()


# Training function
def train(model, train_loader, test_loader, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    train_losses, test_accuracies = [], []
    res_data = {}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        accuracy = evaluate(model, test_loader)
        if (epoch+1)%5 ==0:
            res_data[epoch+1] = {"loss":f'{avg_loss:.4f}', "accuracy": f'{accuracy:.4f}'}
        test_accuracies.append(accuracy)

    plot_metrics(train_losses, test_accuracies)
    print(res_data)

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=GENERAL_CLASSES))
    print("Confusion Matrix:\n", confusion_matrix(targets, preds))

    return accuracy


def explain_with_shap(model, dataset, num_samples=3):
    model.eval()
    background = torch.cat([dataset[i][0].unsqueeze(0) for i in range(10)], dim=0).to(device)
    explainer = shap.GradientExplainer(model, background)
    test_images = torch.cat([dataset[i][0].unsqueeze(0) for i in range(num_samples)], dim=0).to(device)
    shap_values = explainer.shap_values(test_images)

    for i in range(num_samples):
        shap.image_plot([shap_values[0][i]], test_images[i].cpu().numpy().transpose(1,2,0))
