from src.download_kaggle_dataset import download_kaggle_dataset
from src.train import train_model
import os
import argparse


def main():
    # Get the path of the script
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Download and process cricket shot dataset."
    )

    # Define arguments for paths
    parser.add_argument(
        "--input-data-path",
        type=str,
        default=os.path.join(script_path, "data", "raw"),
        help="Path to the input data directory (default: ./data/raw)",
    )
    parser.add_argument(
        "--working-data-path",
        type=str,
        default=os.path.join(script_path, "data", "processed"),
        help="Path to the working data directory (default: ./data/processed)",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=os.path.join(script_path, "results"),
        help="Path to the results directory (default: ./results)",
    )
    parser.add_argument(
        "--checkpoints-path",
        type=str,
        default=os.path.join(script_path, "checkpoints"),
        help="Path to the checkpoints directory (default: ./checkpoints)",
    )

    # Argument for downloading dataset
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        default=False,
        help="Download the dataset (default: False)",
    )

    args = parser.parse_args()

    # Create data directories
    os.makedirs(args.input_data_path, exist_ok=True)
    os.makedirs(args.working_data_path, exist_ok=True)
    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)

    # Download the dataset if specified
    if args.download_dataset:
        if not os.listdir(args.input_data_path):  # Check if the directory is empty
            download_kaggle_dataset(
                "aneesh10/cricket-shot-dataset", download_path=args.input_data_path
            )
        else:
            print(
                f"Dataset already exists in {args.input_data_path}. Skipping download."
            )

    # Train the model
    train_model(args, num_epochs=25, seed=7777)


if __name__ == "__main__":
    main()
