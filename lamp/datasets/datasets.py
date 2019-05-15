from torch.utils import data
import torch
import numpy as np


class SimpleDataset(data.Dataset):

    def __init__(
        self,
        x: torch.tensor,
        labels: torch.tensor
    ):
    
        self.data = x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]