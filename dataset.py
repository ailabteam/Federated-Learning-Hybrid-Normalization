# dataset.py (Corrected Version)
import torch
import numpy as np
from torchvision.datasets import CIFAR10 # type: ignore
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from typing import List, Optional, Tuple

def load_datasets(
    num_clients: int,
    partition_type: str = "iid",
    alpha: float = 0.5,
    batch_size: int = 32,
    data_path: str = "./data"
) -> Tuple[List[DataLoader], DataLoader, Optional[List[DataLoader]]]:
    """
    Loads the CIFAR-10 dataset and partitions it among clients.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    trainset = CIFAR10(data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(data_path, train=False, download=True, transform=transform)

    if partition_type == "non-iid-dirichlet":
        client_train_datasets = _partition_data_dirichlet(trainset, num_clients, alpha)
        client_test_datasets = None
    else: # Default to IID
        client_train_datasets = _partition_data_iid(trainset, num_clients)
        client_test_datasets = _partition_data_iid(testset, num_clients)

    trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2) for ds in client_train_datasets]
    
    testloader = DataLoader(testset, batch_size=batch_size * 2, num_workers=2)
    
    testloaders = None
    if client_test_datasets:
        testloaders = [DataLoader(ds, batch_size=batch_size * 2, num_workers=2) for ds in client_test_datasets]
    
    return trainloaders, testloader, testloaders

def _partition_data_iid(dataset: torch.utils.data.Dataset, num_clients: int) -> List[Subset]:
    """
    Partitions a dataset into IID subsets.
    """
    num_items_per_client = len(dataset) // num_clients
    all_idxs = list(range(len(dataset)))
    np.random.shuffle(all_idxs)
    
    client_subsets = []
    for i in range(num_clients):
        start_idx = i * num_items_per_client
        # The last client gets all remaining samples
        end_idx = (i + 1) * num_items_per_client if i != num_clients - 1 else len(dataset)
        indices = all_idxs[start_idx:end_idx]
        client_subsets.append(Subset(dataset, indices))
        
    return client_subsets

def _partition_data_dirichlet(dataset: torch.utils.data.Dataset, num_clients: int, alpha: float = 0.5) -> List[Subset]:
    """
    Partitions a dataset among clients using a Dirichlet distribution for label skew.
    This version uses a robust index-splitting method.
    """
    if not hasattr(dataset, 'targets'):
        raise ValueError("Dataset must have a 'targets' attribute for Dirichlet partitioning.")

    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    
    proportions = np.random.dirichlet(np.repeat(alpha, num_classes), num_clients)
    
    # Check if any client has zero samples allocated for all classes. 
    # This is rare but can happen with very small alpha and many clients.
    # We add a tiny value to proportions to prevent this, ensuring sum is not zero.
    proportions = proportions + 1e-9

    client_partitions_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        np.random.shuffle(idx_c)
        
        # Calculate the cumulative sum of proportions for the current class
        props_c = proportions[:, c]
        props_c_cumsum = np.cumsum(props_c)
        
        # Normalize the cumulative sum to create split points
        # The last value is guaranteed to be 1.0
        split_points_norm = props_c_cumsum / props_c_cumsum[-1]
        
        # Create split points in the array of indices
        split_points = (len(idx_c) * split_points_norm).astype(int)[:-1]
        
        # Split the indices based on these points
        split_indices = np.split(idx_c, split_points)
        
        for i in range(num_clients):
            client_partitions_indices[i].extend(split_indices[i])

    client_subsets = [Subset(dataset, indices) for indices in client_partitions_indices]
    
    return client_subsets
