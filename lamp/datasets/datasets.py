from torch.utils import data
import torch
import numpy as np


class SimpleDataset(data.Dataset):

    def __init__(
        self,
        x: np.ndarray,
        labels: np.ndarray
    ):
    
        self.data = x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = torch.tensor(
            self.data[index],
            dtype=torch.float
        )
        y = torch.tensor(
            self.labels[index],
            dtype=torch.float
        )

        return X, y