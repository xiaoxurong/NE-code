from data_provider.data_loader import CTGDataset, FHRDataset
from torch.utils.data import DataLoader
import numpy as np

data_dict = {
    "CTG": CTGDataset,
    "FHR": FHRDataset,
    # future classification datasets can go here
}

def data_provider(args, flag):
    """
    Classification-only data provider.
    Returns (dataset, dataloader).
    """

    assert args.data in data_dict, f"Unknown dataset {args.data}"
    DatasetClass = data_dict[args.data]

    # ------------------
    # split selection
    # ------------------
    if flag == "train":
        indices = args.train_idx
        shuffle = True
        drop_last = True
        batch_size = args.batch_size

    elif flag == "val":
        indices = args.val_idx
        shuffle = False
        drop_last = False
        batch_size = args.batch_size

    elif flag == "test":
        indices = args.test_idx
        shuffle = False
        drop_last = False
        batch_size = args.batch_size

    else:
        raise ValueError(f"Unsupported flag: {flag}")

    # ------------------
    # dataset
    # ------------------
    dataset = DatasetClass(
        npz_path=args.root_path + args.data_path,
        indices=indices
    )

    print(f"{flag} samples: {len(dataset)}")

    # ------------------
    # dataloader
    # ------------------
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return dataset, dataloader