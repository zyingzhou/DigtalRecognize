# /usr/bin/env python
# coding: utf-8
# Author:zhiying
# date: 2023.7.12
# description: LeNet pytorch
from torch import nn
import torch


class LeNetPytorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # 定义
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            nn.MaxPool2d(2, 2),
            self.conv2,
            self.relu,
            nn.MaxPool2d(2, 2))

        self.linear = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
