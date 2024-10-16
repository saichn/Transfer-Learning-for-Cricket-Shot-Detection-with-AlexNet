import torch
from torch import nn
from torch.optim import Adam
from src.model import get_model
from src.utils import set_seed, get_data, show_images, plot_curves
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Using device: {device} ---\n")


def run_for_epoch(model, dataloader, criterion, optimizer=None):
    """
    Train or evaluate the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train/evaluate.
        dataloader (DataLoader): DataLoader providing the input data.
        criterion (torch.nn.Module): Loss function to use.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None, evaluation is performed.

    Returns:
        dict: Dictionary containing average loss, accuracy, and lists of correct and incorrect images/labels.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0
    total_samples = len(dataloader.dataset)

    (
        correct_images,
        correct_labels,
        incorrect_images,
        incorrect_predictions,
        incorrect_labels,
    ) = ([], [], [], [], [])

    with torch.set_grad_enabled(is_training):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)

            # Collect correct and incorrect predictions
            correct_mask = predicted_labels == labels
            incorrect_mask = ~correct_mask

            correct_images.extend(inputs[correct_mask].cpu())
            correct_labels.extend(labels[correct_mask].cpu().numpy())

            incorrect_images.extend(inputs[incorrect_mask].cpu())
            incorrect_predictions.extend(predicted_labels[incorrect_mask].cpu().numpy())
            incorrect_labels.extend(labels[incorrect_mask].cpu().numpy())

    return {
        "avg_loss": total_loss / total_samples,
        "accuracy": (len(correct_labels) / total_samples),
        "correct_images": correct_images,
        "correct_labels": correct_labels,
        "incorrect_images": incorrect_images,
        "incorrect_predictions": incorrect_predictions,
        "incorrect_labels": incorrect_labels,
    }


def train_model(args, num_epochs=10, seed=7777):
    """
    Train the AlexNet model with the specified dataset.
    """
    set_seed(seed)

    # Load data
    trainloader, testloader, classes = get_data(
        args.input_data_path, args.working_data_path, seed
    )

    # Display sample training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    show_images(
        images,
        labels,
        classes=classes,
        filename=os.path.join(args.results_path, "sample_training_images.png"),
    )

    # Initialize model, loss function, and optimizer
    model = get_model(out_features=len(classes), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_accuracy = 0.0
    best_model_path = os.path.join(args.checkpoints_path, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch} ---")

        train_results = run_for_epoch(model, trainloader, criterion, optimizer)
        test_results = run_for_epoch(model, testloader, criterion)

        train_losses.append(train_results["avg_loss"])
        train_accuracies.append(train_results["accuracy"])
        test_losses.append(test_results["avg_loss"])
        test_accuracies.append(test_results["accuracy"])

        print(
            f"Train Loss: {train_results['avg_loss']:.4f}, Train Accuracy: {train_results['accuracy']:.4f}"
        )
        print(
            f"Test Loss: {test_results['avg_loss']:.4f}, Test Accuracy: {test_results['accuracy']:.4f}"
        )

        # Save the best model
        if test_results["accuracy"] > best_accuracy:
            best_accuracy = test_results["accuracy"]
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}\n")
            correct_images = test_results["correct_images"]
            correct_labels = test_results["correct_labels"]
            incorrect_images = test_results["incorrect_images"]
            incorrect_labels = test_results["incorrect_labels"]
            incorrect_predictions = test_results["incorrect_predictions"]

    # Display correct and incorrect classifications for best model
    print(
        f"\n--- Correct Predictions made by best model with accuracy {best_accuracy:.4f} ---"
    )
    show_images(
        images=correct_images,
        labels=correct_labels,
        classes=classes,
        filename=os.path.join(
            args.results_path, "best_model_correct_classifications.png"
        ),
    )

    print(
        f"\n--- In-correct Predictions made by best model with accuracy {best_accuracy:.4f} ---"
    )
    show_images(
        images=incorrect_images,
        labels=incorrect_labels,
        predictions=incorrect_predictions,
        classes=classes,
        filename=os.path.join(
            args.results_path, "best_model_incorrect_classifications.png"
        ),
    )

    # Plot loss and accuracy curves
    plot_curves(
        train_losses,
        test_losses,
        "Epochs",
        "Loss",
        "Epochs vs Loss",
        highlight="min",
        filename=os.path.join(args.results_path, "train_and_test_losses.png"),
    )
    plot_curves(
        train_accuracies,
        test_accuracies,
        "Epochs",
        "Accuracy",
        "Epochs vs Accuracy",
        highlight="max",
        filename=os.path.join(args.results_path, "train_and_test_accuracy.png"),
    )
