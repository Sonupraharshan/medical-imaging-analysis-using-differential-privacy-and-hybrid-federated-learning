#!/usr/bin/env python3
import numpy as np
from torch.utils.data import Dataset

class BagDataset(Dataset):
    def __init__(self, folder):
        import os
        from glob import glob
        self.files = sorted(glob(folder + "/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        feats = d['feats'].astype('float32')  # (N, D)
        label = int(d['label'])
        return feats, label
