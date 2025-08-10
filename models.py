# models.py (Full Code)
from torch import nn
import torch.nn.functional as F

class SimpleCNN_BN(nn.Module):
    """A simple CNN for CIFAR-10 with Batch Normalization."""
    def __init__(self, num_classes=10):
        super(SimpleCNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN_GN(nn.Module):
    """A simple CNN for CIFAR-10 with Group Normalization."""
    def __init__(self, num_classes=10):
        super(SimpleCNN_GN, self).__init__()
        # num_groups must be a divisor of num_channels
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.gn1 = nn.GroupNorm(4, 16) # 4 groups, 16 channels
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.gn2 = nn.GroupNorm(8, 32) # 8 groups, 32 channels
        
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # GroupNorm is not typically used for 1D (Linear) layers.
        # LayerNorm or no norm are common choices. We omit it for simplicity.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(norm_type: str = "bn", num_classes: int = 10):
    """
    Returns an instance of the model based on the normalization type.
    
    Args:
        norm_type (str): "bn" for Batch Normalization or "gn" for Group Normalization.
        num_classes (int): Number of output classes.
        
    Returns:
        A PyTorch model instance.
    """
    if norm_type.lower() == "gn":
        # print("Using Group Normalization model.") # Optional: can add noise
        return SimpleCNN_GN(num_classes=num_classes)
    elif norm_type.lower() == "bn":
        # print("Using Batch Normalization model.") # Optional: can add noise
        return SimpleCNN_BN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Choose 'bn' or 'gn'.")
