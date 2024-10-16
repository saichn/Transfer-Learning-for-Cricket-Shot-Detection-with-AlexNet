import urllib.error

import torch.nn as nn
from torchvision import models
import urllib
import torch


def get_model(out_features=None, device="cpu"):
    """
    Load an AlexNet model and modify its classifier for the specific task.

    Args:
        out_features (int): Number of output features/classes for the modified classifier.
        device (str): Device to which the model should be moved ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The modified model ready for training/testing.
    """
    try:
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    except (urllib.error.URLError, Exception) as e:
        print(f"Failed to load model due to: {e}. Trying alternative approach")
        model = torch.hub.load("pytorch/vision", "alexnet", pretrained=True)

    # Modify classifier to match the number of classes
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features, out_features=out_features
    )

    # Freeze feature layers if using pretrained weights
    for param in model.features.parameters():
        param.requires_grad = False

    return model.to(device)
