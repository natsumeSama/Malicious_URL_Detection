"""
Data loading and preprocessing utilities for malicious URL classification.

This module provides:
- Byte-level URL encoding (UTF-8 → bytes → fixed-length tensor)
- A PyTorch Dataset that yields (url_tensor, label_idx)
- Train/val/test DataLoaders with reproducible splits
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------- utils ----------------


def get_num_workers() -> int:
    """
    Pick a sensible default for DataLoader workers.

    Returns:
        int: 0 on Windows (to avoid multiprocessing issues), else CPU count.
    """
    return 0 if os.name == "nt" else os.cpu_count()


def get_classes(df: pd.DataFrame) -> dict:
    """
    Build a stable label→index mapping from the 'type' column.

    Args:
        df: DataFrame with a 'type' column of string class names.

    Returns:
        dict[str,int]: e.g., {'benign': 0, 'defacement': 1, ...}
    """
    classes = sorted(df["type"].unique().tolist())
    return {label: idx for idx, label in enumerate(classes)}


def encode_url(url: str, max_len: int) -> np.ndarray:
    """
    Encode a URL string to a fixed-length array of byte IDs (0..255).

    Steps:
      1) UTF-8 encode the string
      2) Truncate to max_len if longer
      3) Pad with zeros to max_len if shorter

    Args:
        url: URL string.
        max_len: Desired fixed length.

    Returns:
        np.ndarray (uint8) of shape (max_len,).
    """
    b = url.encode("utf-8", errors="replace")
    arr = np.frombuffer(b, dtype=np.uint8).copy()  # copy => writable buffer
    n = arr.size
    if n >= max_len:
        return arr[:max_len]
    out = np.zeros(max_len, dtype=np.uint8)
    out[:n] = arr
    return out


# ------------- dataset -----------------

MAX_LEN = 300
NUM_WORKERS = get_num_workers()


class StringFolderCustom(Dataset):
    """
    PyTorch Dataset for malicious URL classification.

    Each __getitem__ returns:
      - url_tensor: LongTensor of shape [max_len] (byte IDs)
      - class_idx : LongTensor scalar with the label index
    """

    def __init__(self, path: str | Path, max_len: int = MAX_LEN):
        """
        Args:
            path: CSV file path containing 'url' and 'type' columns.
            max_len: Fixed sequence length for encoded URLs.
        """
        self.data = pd.read_csv(path, usecols=["url", "type"])
        self.max_len = max_len

        # Build label mapping once
        self.class_to_idx = get_classes(self.data)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Precompute numeric labels for speed
        self.label_idx = (
            self.data["type"].map(self.class_to_idx).astype("int64").to_numpy()
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # URL → bytes → pad/trunc → LongTensor
        url_bytes = encode_url(self.data["url"].iloc[index], self.max_len)
        url_tensor = torch.from_numpy(url_bytes).long()

        # Label LongTensor (scalar)
        class_idx = torch.tensor(self.label_idx[index], dtype=torch.long)

        return url_tensor, class_idx


def create_dataloaders(
    dir: str | Path,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with reproducible splits.

    Split strategy:
      - 80% train+val / 20% test
      - From the 80% train+val, reserve 20% for val
        => final: 64% train, 16% val, 20% test

    Args:
        dir: Path to the CSV (or path passed to the dataset).
        batch_size: Batch size for the loaders.
        num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = StringFolderCustom(path=dir)

    # 80% train+val / 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=g)

    # From train_data, split 80/20 => 64% train, 16% val overall
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size], generator=g)

    pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        drop_last=True,  # often helpful for stable batch shapes
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir.parent / "data" / "db" / "malicious_phish1.csv"

    ds = StringFolderCustom(csv_path, max_len=20)
    print("Dataset length:", len(ds))

    url_t, label_t = ds[0]
    print("URL tensor shape:", url_t.shape)
    print("First 10 bytes:", url_t[:10].tolist())
    print("Label index:", label_t.item())
    print("Label name:", ds.idx_to_class[label_t.item()])
