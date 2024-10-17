import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split


def set_seed(seed):
    """
    Set random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unnormalize(img, mean, std):
    """
    Unnormalize a tensor image to display it properly.

    Args:
        img (tensor): The input image tensor to unnormalize.
        mean (float): Mean used during normalization.
        std (float): Standard deviation used during normalization.

    Returns:
        np.array: Unnormalized image in NumPy array format.
    """
    return img.detach().cpu().numpy() * std + mean


def show_images(
    images, labels, predictions=None, classes=None, shuffle=True, filename=None
):
    """
    Display a grid of images with their corresponding labels and optional predictions.

    Args:
        images (list): List of images to display.
        labels (list): List of true labels for the images.
        predictions (list, optional): List of predicted labels, to display incorrect predictions in red.
        classes (list): List of class names corresponding to label indices.
        shuffle (bool): Whether to shuffle the images for display. Defaults to True.
        filename (str, optional): Path to save the image grid. If None, the images are displayed instead of saved.
    """
    num_of_images_to_show = 9
    if shuffle:
        indices = random.sample(range(0, len(labels)), k=num_of_images_to_show)
    else:
        indices = range(num_of_images_to_show)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    for axes_idx, img_idx in enumerate(indices):
        img, label = images[img_idx], labels[img_idx]
        img = unnormalize(img, 0.5, 0.5).transpose(1, 2, 0)
        axes[axes_idx].imshow(img)
        if predictions is not None and predictions[img_idx] != label:
            title = f"Pred: {classes[predictions[img_idx]]} | True: {classes[label]}"
            color = "red"
        else:
            title = f"{classes[label]}"
            color = "green"

        axes[axes_idx].set_title(title, color=color)
        axes[axes_idx].axis("off")

    plt.tight_layout()

    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Saved image grid to {filename}")
    else:
        plt.show()


def plot_curves(
    train_data, test_data, xlabel, ylabel, title, highlight=None, filename=None
):
    """
    Plot training and test data curves (loss or accuracy) over epochs.

    Args:
        train_data (list): Data for training set (loss or accuracy).
        test_data (list): Data for test set (loss or accuracy).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        highlight (str or None): 'max' to highlight the highest value, 'min' to highlight the lowest value,
                                 or None for no highlight.
        filename (str, optional): Path to save the image grid. If None, the images are displayed instead of saved.
    """
    plt.figure()

    # Plot training and test curves
    plt.plot(train_data, label="Train")
    plt.plot(test_data, label="Test")

    if highlight:
        # Determine whether to highlight the highest or lowest value
        if highlight == "max":
            best_epoch = np.argmax(test_data)
            best_value = test_data[best_epoch]
        elif highlight == "min":
            best_epoch = np.argmin(test_data)
            best_value = test_data[best_epoch]
        else:
            raise ValueError("Invalid highlight flag. Use 'max', 'min', or None.")

        # Highlight the best value point
        plt.scatter(
            best_epoch,
            best_value,
            color="green",
            label=f"Best Model (Epoch: {best_epoch+1}, {ylabel}: {best_value:.4f})",
        )

        # Label the point with "best model"
        plt.text(
            best_epoch,
            best_value,
            f"Best Model\n({best_value:.4f})",
            ha="center",
            va="bottom",
            color="green",
        )

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # Set x-axis ticks to integers (epochs)
    plt.xticks(ticks=range(len(train_data)), labels=range(1, len(train_data) + 1))

    # Add legend and display the plot
    plt.legend()

    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Saved image grid to {filename}")
    else:
        plt.show()


def create_symlinks(src_path, dst_path, label, images):
    """
    Create symbolic links for images to organize data into train and test sets.

    Args:
        src_path (str): The source directory containing original images.
        dst_path (str): The destination directory to create symbolic links.
        label (str): The class label of the images.
        images (list): List of image filenames to create symbolic links for.
    """
    os.makedirs(os.path.join(dst_path, label), exist_ok=True)
    for img in images:
        src = os.path.join(src_path, label, img)
        dst = os.path.join(dst_path, label, img)
        if not os.path.islink(dst):
            os.symlink(src, dst)

def get_transform():
    """
    Returns a composition of image transformations to be applied before inputting the image to the model.

    The transformations include:
    - Resizing the image to a fixed size of 224x224 pixels (the standard input size for AlexNet).
    - Converting the image to a PyTorch tensor.
    - Normalizing the pixel values to have a mean of [0.5, 0.5, 0.5] and a standard deviation of [0.5, 0.5, 0.5]
      across the RGB channels, which helps to scale the image to a standard range suitable for the pre-trained model.

    Returns:
        torchvision.transforms.Compose: A composition of transformations to be applied to the input images.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

def get_data(input_data_path, working_data_path, seed):
    """
    Load and split dataset into training and testing sets, then apply necessary transformations.

    Args:
        input_data_path (str): Path to the original dataset.
        working_data_path (str): Path where train/test split data will be stored.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Train loader, test loader, and list of class names.
    """
    train_path = os.path.join(working_data_path, "train")
    test_path = os.path.join(working_data_path, "test")

    os.makedirs(working_data_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    classes = sorted(os.listdir(input_data_path))
    print(f"Classes: {classes}\n")

    # Split dataset into train and test
    for label in classes:
        images = os.listdir(os.path.join(input_data_path, label))
        train_images, test_images = train_test_split(
            images, test_size=0.2, random_state=seed
        )
        create_symlinks(input_data_path, train_path, label, train_images)
        create_symlinks(input_data_path, test_path, label, test_images)

    transform = get_transform()

    train_data = datasets.ImageFolder(train_path, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)

    test_data = datasets.ImageFolder(test_path, transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=32)

    return trainloader, testloader, classes
