# face_graph_dataset.py
from pathlib import Path
import random
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from collections import Counter
import numpy as np


class FaceGraphDataset(Dataset):
    """
    Expects:
      graph_data/
        my_face/*.pt
        not_my_face/*.pt

    Each .pt is a torch_geometric.data.Data with fields: x, edge_index
    We'll attach y (label) here based on the folder name.
    """
    def __init__(self, root: str = "face_features"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Could not find graph root: {self.root.resolve()}")

        self.class_names = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.class_names:
            raise RuntimeError(f"No class subfolders in {self.root.resolve()}")

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for c in self.class_names:
            for pt in (self.root / c).glob("*.pt"):
                self.samples.append((pt, self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No .pt graphs found under {self.root.resolve()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        data: Data = torch.load(path, weights_only=False)
        data.y = torch.tensor([y], dtype=torch.long)  # graph label
        return data


def split_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    ):
    g = torch.Generator()
    g.manual_seed(42)
    
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]
    test_idx  = indices[n_train+n_val:]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_idx)
    test_subset  = torch.utils.data.Subset(dataset, test_idx)


    #labels = np.array([dataset.samples[i][1] for i in train_idx])
    labels = np.array([dataset.samples[i][1] for i in train_idx])

    train_loader = PyGDataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_subset, batch_size=batch_size*2, shuffle=False)
    test_loader  = PyGDataLoader(test_subset, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader, labels

if __name__ == "__main__":
    ds = FaceGraphDataset("graph_data")
    print("Classes:", ds.class_names)
    print("Num samples:", len(ds))

    train_loader, val_loader, test_loader = split_loaders(ds, batch_size=32)

    batch = next(iter(train_loader))
    print("\nOne batch:")
    print("x:", batch.x.shape)
    print("edge_index:", batch.edge_index.shape)
    print("y:", batch.y.shape, "labels:", batch.y.view(-1)[:10].tolist())
    print("num_graphs:", batch.num_graphs)



    ys = [y for _, y in ds.samples]
    print("Label counts:", Counter(ys))
    print("class_to_idx:", ds.class_to_idx)

