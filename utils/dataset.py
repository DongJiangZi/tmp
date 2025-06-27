# -*- coding = utf-8 -*-
# @File : dataset.py
# @Software : PyCharm
import os
import ast
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SSLECGTextDataset(Dataset):
    def __init__(self, csv_path, data_dir, split='train', val_ratio=0.2, random_seed=42):
        super(SSLECGTextDataset, self).__init__()
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['scp_codes'].notna()]
        self.df['scp_codes'] = self.df['scp_codes'].apply(ast.literal_eval)

        indices = list(self.df.index)
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=random_seed)

        if split == 'train':
            self.df = self.df.loc[train_idx]
        elif split == 'val':
            self.df = self.df.loc[val_idx]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_rel_path = row['filename_hr'] + '.dat'
        ecg_path = os.path.join(self.data_dir, ecg_rel_path)

        try:
            with open(ecg_path, 'rb') as f:
                raw = torch.tensor(bytearray(f.read()), dtype=torch.uint8).float().view(-1)
        except Exception as e:
            print(f"Failed to load: {ecg_path} due to {e}")
            return self.__getitem__((idx + 1) % len(self))

        if raw.numel() % 16 != 0:
            return self.__getitem__((idx + 1) % len(self))

        ecg = raw.view(16, -1)
        text = ", ".join(row['scp_codes'].keys())
        return ecg, text


class ZeroShotTestECGTextDataset(Dataset):
    def __init__(self, csv_path, data_dir, categories):
        super(ZeroShotTestECGTextDataset, self).__init__()
        self.data_dir = data_dir
        self.categories = categories
        self.label2idx = {label: i for i, label in enumerate(categories)}

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['scp_codes'].notna()]
        self.df['scp_codes'] = self.df['scp_codes'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_rel_path = row['filename_hr'] + '.dat'
        ecg_path = os.path.join(self.data_dir, ecg_rel_path)

        try:
            with open(ecg_path, 'rb') as f:
                raw = torch.tensor(bytearray(f.read()), dtype=torch.uint8).float().view(-1)
        except Exception as e:
            print(f"Failed to load: {ecg_path} due to {e}")
            return self.__getitem__((idx + 1) % len(self))

        if raw.numel() % 16 != 0:
            return self.__getitem__((idx + 1) % len(self))

        ecg = raw.view(16, -1)
        labels = list(row['scp_codes'].keys())
        label_index = self.label2idx.get(labels[0], 0)
        return ecg, torch.tensor(label_index, dtype=torch.long)

    def load_categories(self):
        return self.categories
