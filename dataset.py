from torchvision import datasets, transforms
import torch
from const import BATCH_SIZE, NWORKERS
from abc import ABC, abstractmethod
from typing import Tuple


def get_mean_and_std(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean and standard deviation for given data loader"""
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std

class Dataset(ABC):
    @abstractmethod
    def get_train_loader(self) -> torch.utils.data.DataLoader:
        pass

    @abstractmethod
    def get_val_loader(self) -> torch.utils.data.DataLoader:
        pass


class FashionMnist:
    def __init__(self, path: str) -> None:
        print("Using FashionMNIST")
        base_transforms = transforms.Compose([transforms.ToTensor()])
        stats_trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=base_transforms
        )
        stats_train_loader = torch.utils.data.DataLoader(
            stats_trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NWORKERS
        )

        mean, std = get_mean_and_std(stats_train_loader)
        mean = mean.numpy()[0]
        std = std.numpy()[0]

        print("Trainset mean: ", mean)
        print("Trainset std: ", std)

        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1)),
                transforms.Normalize((mean,), (std,)),
                transforms.RandomErasing(),
            ]
        )
        val_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
        )

        trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=train_transforms
        )
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS
        )
        valset = datasets.FashionMNIST(path, train=False, transform=val_transforms)
        self.val_loader = torch.utils.data.DataLoader(
            valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NWORKERS
        )

    def get_train_loader(self) -> torch.utils.data.DataLoader:
        return self.train_loader
    
    def get_val_loader(self) -> torch.utils.data.DataLoader:
        return self.val_loader
