import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from data_loader.data_loader import PetsDataset


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        h1 = 16
        h2 = 32
        kernel = (5, 5)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=h1,
                kernel_size=kernel, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(
                in_channels=h1, out_channels=h2,
                kernel_size=kernel, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.linear = nn.Sequential(
            nn.Linear(107648, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, X):
        out = self.conv(X)
        out = out.reshape(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    dataset = PetsDataset(
        root_dir=f'{os.getcwd()}/dataset/valid',
        transform=transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
        ])
    )

    X, y = dataset[10]

    # plt.imshow(X.permute(1, 2, 0))  # important don't use reshape
    # plt.show()
    #
    # plt.imshow(y.numpy().squeeze())
    # plt.show()

    dataloader = DataLoader(
        dataset, batch_size=4,
        shuffle=True, num_workers=0
    )

    net = CNN_Model()
    X_batch, y_batch = next(iter(dataloader))

    print(X_batch.shape)
    out = net(X_batch)
    print(out.shape)

