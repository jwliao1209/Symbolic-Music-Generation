import random

import torch
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    def __init__(self, data_list, max_length=512):
        self.data_list = data_list
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]["tokens"]
        start = random.randint(0, len(data) - self.max_length)
        return {
            "input_ids": torch.tensor(data[start : start + self.max_length]),
            "labels": torch.tensor(data[start : start + self.max_length]),
        }

    def get_loader(self, batch_size, shuffle, num_workers=1, pin_memory=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
