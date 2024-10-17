import os
import torch
from torchvision import datasets
import torch.nn as nn
from src.model import get_model
from src.utils import get_transform
import matplotlib.pyplot as plt


def run_evaluation(model, data_path, device):
    """
    Evaluate the given model on the dataset located at data_path.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_path (str): The path to the dataset to evaluate on.
        device (str): The device to use for computation ('cpu' or 'cuda').

    Returns:
        tuple: Average loss and accuracy of the model on the dataset.
    """
    model = model.to(device)
    transform = get_transform()

    # Load the dataset and create a DataLoader
    data = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    correct_predictions = 0
    total_loss = 0
    total_samples = len(dataloader.dataset)

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def plot_results(model_names, accuracies, output_path):
    """
    Plots a comparison bar chart of model accuracies.

    Args:
        model_names (list): List of model names.
        accuracies (list): List of accuracies corresponding to the models.
        output_path (str): Path to save the plot.
    """
    plt.figure()
    plt.bar(model_names, accuracies, color=['green', 'orange'], width=0.2)
    plt.ylim(0, 1)
    for index, value in enumerate(accuracies):
        plt.text(index, value - 0.05, f"{value * 100:.2f}%", ha='center', fontsize=10, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.title('Cricket Shot Model vs ImageNet Model Accuracy Comparison')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)


def evaluate_models(script_path, device):
    """
    Evaluates both the Cricket Shot and ImageNet models on the test dataset
    and generates accuracy comparison plot.
    """
    test_data_path = os.path.join('data', 'processed', 'test')

    # Evaluate the cricket-shot-trained model
    cricket_shot_model = get_model(out_features=4, device=device, use_cricket_shot_model=True)
    avg_loss_cricket_test, accuracy_cricket_test = run_evaluation(cricket_shot_model, test_data_path, device)

    # Evaluate the default ImageNet-pretrained model
    imagenet_model = get_model(out_features=4, device=device, use_cricket_shot_model=False)
    avg_loss_imagenet_test, accuracy_imagenet_test = run_evaluation(imagenet_model, test_data_path, device)

    # Plot and save the results
    model_names = ['Cricket Shot Model', 'ImageNet Model']
    accuracies = [accuracy_cricket_test, accuracy_imagenet_test]
    output_path = os.path.join(script_path, 'results', 'cricket_shot_model_vs_imagenet_model_accuracy_comparison.png')
    plot_results(model_names, accuracies, output_path)

    print(f"Avg loss (Cricket Shot Model): {avg_loss_cricket_test:.4f}. Accuracy: {accuracy_cricket_test:.4f}")
    print(f"Avg loss (ImageNet Model): {avg_loss_imagenet_test:.4f}. Accuracy: {accuracy_imagenet_test:.4f}")


if __name__ == "__main__":
    evaluate_models()

