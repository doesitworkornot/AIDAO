import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scripts.data_utils import get_connectome


def add_noise(data, noise_factor=0.08):
    noise = torch.randn_like(data) * noise_factor
    return data + noise


def mask_data(data, mask_probability=0.005):
    mask = torch.rand(data.size()) > mask_probability
    return data * mask


def scale_data(data, scale_factor_range=(0.93, 1.07)):
    scale = np.random.uniform(*scale_factor_range)
    return data * scale


def jitter(data, jitter_factor=0.005):
    jitter = torch.randn_like(data) * jitter_factor
    return data + jitter


class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, augmentations=None):
        self.data = data
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # Применяем аугментации, если они указаны
        if self.augmentations:
            for aug in self.augmentations:
                x = aug(x)

        return x, y


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (bs, 32, 419, 419)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (bs, 32, 209, 209)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (bs, 64, 209, 209)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (bs, 64, 104, 104)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (bs, 128, 104, 104)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (bs, 128, 52, 52)
        )


        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 52 * 52, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # (bs, 128 * 52 * 52)
        x = self.fc_layers(x)
        return x



bnu_series_path = '../data/ts_cut/HCPex/bnu.npy'
bnu_labels_path = '../data/ts_cut/HCPex/bnu.csv'
ihb_series_path = '../data/ts_cut/HCPex/ihb_old.npy'
ihb_labels_path = '../data/ts_cut/HCPex/ihb.csv'

X_bnu = np.load(bnu_series_path)
print(X_bnu.shape)
Y_bnu = pd.read_csv(bnu_labels_path)
print(Y_bnu.shape)
X_ihb = np.load(ihb_series_path)
print(X_ihb.shape)
Y_ihb = pd.read_csv(ihb_labels_path)
print(Y_ihb.shape)

X_bnu = get_connectome(X_bnu)
X_ihb = get_connectome(X_ihb)

X = np.concatenate([X_bnu, X_ihb])
Y = np.concatenate([Y_bnu, Y_ihb])

x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.15, random_state=10)


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_train_tensor = x_train_tensor.unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

x_validate_tensor = torch.tensor(x_validate, dtype=torch.float32)
x_validate_tensor = x_validate_tensor.unsqueeze(1)
y_validate_tensor = torch.tensor(y_validate, dtype=torch.long)

augmentations = [add_noise, jitter, scale_data, mask_data]


train_dataset = VectorDataset(x_train_tensor, y_train_tensor, augmentations=augmentations)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

validate_dataset = VectorDataset(x_validate_tensor, y_validate_tensor)
validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=False)

input_size = x_train.shape[1]
output_size = len(np.unique(Y))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validate_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the validation set: {accuracy:.2f}%')

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
    validate()

model = model.to('cpu')
torch.save(model.state_dict(), 'conv_net_aug.pth')
