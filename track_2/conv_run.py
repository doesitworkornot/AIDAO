import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.data_utils import get_connectome

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

        x = x.view(x.size(0), -1)  

        x = self.fc_layers(x)
        return x


X = np.load('./data/ts_cut/HCPex/predict.npy')
print(X.shape)
X = get_connectome(X)
x_tensor = torch.tensor(X, dtype=torch.float32)
x_tensor = x_tensor.unsqueeze(1)
inf_loader = DataLoader(x_tensor, batch_size=4, shuffle=False)

model = ConvNet()
model.load_state_dict(torch.load('conv_net_aug.pth', weights_only=True))

y_pred = torch.empty(x_tensor.size(0))
for i, inputs in enumerate(inf_loader):
    pred = model(inputs)
    if pred.dim() > 1:
        pred = torch.argmax(pred, dim=1)
    start_index = i * inf_loader.batch_size
    end_index = start_index + pred.size(0)
    y_pred[start_index:end_index] = pred

print(y_pred)

solution = pd.DataFrame(data=y_pred, columns=['prediction'])
solution.to_csv('./solution.csv', index=False)
