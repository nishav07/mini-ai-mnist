from typing import Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

def get_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    transform = get_transforms()

    full_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size


    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
