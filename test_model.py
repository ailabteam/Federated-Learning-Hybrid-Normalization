# test_model.py
import torch
from resnet import ResNet18

print("--- Testing BN model (default) ---")
model_bn = ResNet18()
print(model_bn)

print("\n--- Testing HypeNorm model ---")
# 2 layers of PopulationNorm, 2 layers of BatchNorm
norm_config = ['pn', 'pn', 'bn', 'bn']
model_hybrid = ResNet18(norm_types=norm_config)
print(model_hybrid)

# Test forward pass
dummy_input = torch.randn(2, 3, 32, 32) # CIFAR-10 image size
try:
    output = model_hybrid(dummy_input)
    print("\nForward pass successful!")
    print("Output shape:", output.shape)
except Exception as e:
    print("\nForward pass failed:", e)
