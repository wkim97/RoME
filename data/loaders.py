"""Dataset loaders. Images stay in ``[0, 1]``; no normalisation is applied."""

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = ['build_loader']

_IMAGENET_DIR = {'imagenet100': 'imagenet_100', 'imagenet1k': 'imagenet'}


def _cifar10_transforms(train):
    if train:
        return transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])


def _imagenet_transforms(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def _build_dataset(dataset, data_dir, train):
    if dataset == 'cifar10':
        return datasets.CIFAR10(data_dir, train=train, download=True,
                                transform=_cifar10_transforms(train))
    split_dir = os.path.join(data_dir, _IMAGENET_DIR[dataset], 'train' if train else 'val')
    return datasets.ImageFolder(split_dir, transform=_imagenet_transforms(train))


def build_loader(dataset, data_dir, train, batch_size, num_workers=8,
                 num_examples=0, seed=0):
    """``num_examples`` caps the split size: the first ``num_examples`` samples
    when ``train``, a seeded random subset otherwise. ``0`` uses everything."""
    data = _build_dataset(dataset, data_dir, train)

    if num_examples and num_examples < len(data):
        if train:
            indices = range(num_examples)
        else:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(data), generator=generator)[:num_examples].tolist()
        data = Subset(data, indices)

    return DataLoader(data, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True, drop_last=train)
