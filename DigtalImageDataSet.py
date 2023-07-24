# /usr/bin/env python
# coding:utf-8
# custom dataset

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, label_path, transform=None, target_transform=None):
        self.data_list = []
        with open(label_path, encoding='utf-8') as f:
            for line in f.readlines():
                image_path, label = line.split('\t')
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        image = read_image(image_path)
        image = torch.tensor(image, dtype=torch.float32)
        label = int(label)
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            image = self.transform()
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
