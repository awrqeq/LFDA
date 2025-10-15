import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size: int, train: bool, path: str, num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalization is important for many models, but for visualization and LPIPS,
        # it's often easier to work with tensors in the [0, 1] range.
        # We will apply normalization inside the training loop if needed by the model.
        # For now, let's keep it simple.
    ])

    dataset = datasets.CIFAR10(root=path, train=train, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)

    return dataloader
