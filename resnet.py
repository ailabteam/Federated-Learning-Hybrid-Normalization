# resnet.py
import torch
import torch.nn as nn
from typing import Type, Union, List, Optional

class PopulationNorm2d(nn.Module):
    """
    Population Normalization Layer.
    This layer receives global mean and variance from the server
    and uses them to normalize the input. It has its own learnable
    affine parameters (gamma and beta).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(PopulationNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum # Not used for normalization, but kept for compatibility
        
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Global stats (non-learnable, updated by the server)
        self.register_buffer('population_mean', torch.zeros(num_features))
        self.register_buffer('population_var', torch.ones(num_features))
        
    def forward(self, x):
        # Reshape for broadcasting
        pop_mean = self.population_mean.view(1, -1, 1, 1)
        pop_var = self.population_var.view(1, -1, 1, 1)
        
        # Normalize using population statistics
        x_normalized = (x - pop_mean) / (torch.sqrt(pop_var + self.eps))
        
        # Apply learnable affine parameters
        return self.weight.view(1, -1, 1, 1) * x_normalized + self.bias.view(1, -1, 1, 1)

def get_norm_layer(norm_type: str, num_features: int) -> nn.Module:
    """Factory function to get a normalization layer."""
    if norm_type == 'bn':
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'pn':
        return PopulationNorm2d(num_features)
    elif norm_type == 'gn':
        # Ensure num_groups is a divisor of num_features
        num_groups = 8 if num_features % 8 == 0 else (4 if num_features % 4 == 0 else 2)
        return nn.GroupNorm(num_groups, num_features)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

class BasicBlock(nn.Module):
    """Basic Block for ResNet18"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm_type, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm_type, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                get_norm_layer(norm_type, self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_types: Optional[List[str]] = None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        if norm_types is None:
            # Default to all Batch Normalization if not specified
            norm_types = ['bn'] * 4
        if len(norm_types) != 4:
            raise ValueError("norm_types list must have 4 elements, one for each ResNet layer.")
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm_types[0], 64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_type=norm_types[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_type=norm_types[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_type=norm_types[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_type=norm_types[3])
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, norm_types: Optional[List[str]] = None):
    """
    Constructs a ResNet-18 model.
    
    Args:
        num_classes (int): Number of classes for the final layer.
        norm_types (list of str): A list of 4 strings specifying the norm layer
                                  for each of the 4 ResNet layers. 
                                  Example: ['pn', 'pn', 'bn', 'bn']
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_types=norm_types)
