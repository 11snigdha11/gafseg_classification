# utils/cifar100_dataset.py

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def dirichlet_split_noniid(labels, n_clients, alpha=0.1):

    n_classes = np.max(labels) + 1

    client_idxs = [[] for _ in range(n_clients)]

    for c in range(n_classes):

        idx_c = np.where(labels == c)[0]

        np.random.shuffle(idx_c)

        proportions = np.random.dirichlet(
            alpha=np.ones(n_clients) * alpha
        )

        proportions = (
            np.cumsum(proportions) * len(idx_c)
        ).astype(int)[:-1]

        split_idxs = np.split(idx_c, proportions)

        for client_id in range(n_clients):
            client_idxs[client_id].extend(
                split_idxs[client_id]
            )

    return client_idxs
def get_cifar100_loaders(args):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    trainset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    testset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    # nonIID split
    labels = np.array(trainset.targets)

    client_indices = dirichlet_split_noniid(
        labels,
        args.num_clients,
        alpha=0.1
    )

    train_loaders = []

    for i in range(args.num_clients):

        print(f"Client {i}: {len(client_indices[i])} samples")

        client_ds = Subset(
            trainset,
            client_indices[i]
        )

        train_loaders.append(
            DataLoader(
                client_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True
            )
        )

    test_loader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )

    return train_loaders, test_loader