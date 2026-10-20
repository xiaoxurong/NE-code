import numpy as np
import torch
from torch.utils.data import Dataset

class CTGDataset(Dataset):
    def __init__(self, npz_path, indices=None):
        """
        Parameters
        ----------
        npz_path : str
            Path to a window file written by scripts/prepare_ctu_uhb.py
        indices : np.ndarray or list, optional
            Indices of samples to use (for train/val/test split)
        """
        data = np.load(npz_path)

        X = data["X"].transpose(0, 2, 1)            # CTG traces with shape (N, 2, 7200)
        y = data["y"]            # shape (N,)
        subject_ids = data["subject_ids"]

        if indices is not None:
            X = X[indices]
            y = y[indices]
            subject_ids = subject_ids[indices]

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.subject_ids = subject_ids  # keep as numpy array

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
