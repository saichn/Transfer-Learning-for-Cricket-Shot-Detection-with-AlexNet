import os
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_transform
from src.model import get_model

def load_image(image_source):
    """
    Load an image from a URL or a local file path.

    Args:
        image_source (str): URL or file path of the image.

    Returns:
        image (PIL.Image.Image): The loaded image.
    """
    # Check if the image source is a URL
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Failed to retrieve image from URL. Status code: {response.status_code}")
    elif os.path.isfile(image_source):
        image = Image.open(image_source)
    else:
        raise Exception(f"Invalid image source: {image_source}. Please provide a valid URL or file path.")

    return image

def predict_shot(image, classes, device):
    """
        Predict the cricket shot type from an input image using a trained model.

        Args:
            image (str or PIL.Image): The path to the image file or a PIL image object.
                                       Can be a local file or a URL.
            classes (list of str): List of class names corresponding to the model's output.
            device (str or torch.device): The device to use for computation (e.g., "cpu" or "cuda").

        Returns:
            None: The function does not return any value. It displays the predicted class
                  probabilities and shows the input image with the top prediction.

        This function loads the image, applies necessary transformations,
        and uses a trained cricket shot model to predict the type of shot.
        It outputs the class probabilities and visualizes the top prediction.
    """
    # Load the image from the URL or local file
    image = load_image(image)

    # Apply transformations
    transform = get_transform()
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Load the model (Cricket Shot Model)
    cricket_shot_model = get_model(out_features=4, device=device, use_cricket_shot_model=True)

    # Set the model to evaluation mode
    cricket_shot_model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get the output predictions from the model (logits)
        output = cricket_shot_model(transformed_image.to(device))

        # Apply softmax to get prediction probabilities
        probabilities = F.softmax(output, dim=1)

        # Sort the probabilities in descending order
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)

        # Convert tensors to numpy for easier handling
        sorted_probabilities = sorted_probabilities[0].cpu().numpy()
        sorted_indices = sorted_indices[0].cpu().numpy()

        # Display the probabilities and class names
        print("Class probabilities:")
        for i, idx in enumerate(sorted_indices):
            label = classes[idx]
            prob = sorted_probabilities[i]
            print(f"{label}: {prob * 100:.2f}%")

            # For the top prediction, display the image with the class and probability
            if i == 0:
                plt.figure(figsize=(5, 5))
                plt.imshow(np.array(image))
                plt.title(f"Prediction: {label} (Prob: {prob * 100:.2f}%)", color='green')
                plt.axis("off")

    # Show the plot
    plt.show()
