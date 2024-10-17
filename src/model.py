import torch
from torchvision import models
import torch.nn as nn
import urllib
import os
import requests


def get_model(out_features=None, device="cpu", use_cricket_shot_model=False):
    """
    Load an AlexNet model, either the default pre-trained on ImageNet or a fine-tuned
    model trained on a cricket-shot dataset, and modify its classifier for the specific task.

    Args:
        out_features (int): Number of output features/classes for the modified classifier.
        device (str): Device to which the model should be moved ('cpu' or 'cuda').
        use_cricket_shot_model (bool): If True, loads the best model trained on the cricket-shot dataset.

    Returns:
        model (torch.nn.Module): The modified model ready for training/testing.
    """
    # If user wants to use the cricket-shot model, download and load it
    if use_cricket_shot_model:
        # Download the cricket-shot model if not already available
        url = "https://github.com/saichn/Transfer-Learning-for-Cricket-Shot-Detection-with-AlexNet/releases/download/v1.0.0/best_model.pth"
        model_path = "best_model.pth"

        if os.path.isfile(model_path):
            print(f"{model_path} exists. Skipping download...")
        else:
            response = requests.get(url)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                print("Cricket-shot model weights downloaded successfully.")
            else:
                raise Exception(f"Failed to download model weights: {response.status_code}")

        # Load the AlexNet model and apply the custom weights
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=out_features)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded cricket-shot model.")

    else:
        # Load default AlexNet pretrained on ImageNet
        try:
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        except (urllib.error.URLError, Exception) as e:
            print(f"Failed to load pretrained model due to: {e}. Trying alternative approach")
            model = torch.hub.load("pytorch/vision", "alexnet", pretrained=True)

        # Modify the classifier to match the number of classes
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_features=out_features)

        # Freeze the feature extractor layers if using pretrained weights
        for param in model.features.parameters():
            param.requires_grad = False

        print("Loaded ImageNet-pretrained model.")

    return model.to(device)

