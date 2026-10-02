import numpy as np
import torch
from torch.utils.data import Dataset

class CTGDataset(Dataset):
    def __init__(self, npz_path, indices=None):
        """
        Parameters
        ----------
        npz_path : str
            Path to CTG30minSegments.npz
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

# process univariate FHR data into (N, 1, T) shape for consistency with multivariate CTG data
class FHRDataset(Dataset):
    def __init__(self, npz_path, indices=None):
        """
        Parameters
        ----------
        npz_path : str
            Path to FHR10minSegments1Chan.npz
        indices : np.ndarray or list, optional
            Indices of samples to use (for train/val/test split)
        """
        data = np.load(npz_path)
        print(f"Loaded FHR dataset with X shape {data['X'].shape}, y shape {data['y'].shape}, subject_ids shape {data['subject_ids'].shape}")
        print("Any NaN in X?", np.isnan(data['X']).any())

        X = torch.tensor(data["X"]).unsqueeze(1)          # FHR traces with shape (N, 1, 2400)
        y = data["y"]            # shape (N,)
        subject_ids = data["subject_ids"]

        if indices is not None:
            X = X[indices]
            y = y[indices]
            subject_ids = subject_ids[indices]

        self.X = X.float()
        self.y = torch.from_numpy(y).long()
        self.subject_ids = subject_ids  # keep as numpy array

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    