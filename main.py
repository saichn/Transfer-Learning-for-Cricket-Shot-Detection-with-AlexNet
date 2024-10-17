from src.download_kaggle_dataset import download_kaggle_dataset
from src.train import train_model
from src.predict import predict_shot
from src.evaluate import evaluate_models
import os
import argparse
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Using device: {device} ---\n")

def main():
    # Get the path of the script
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Download and process cricket shot dataset."
    )

    # Define subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand for training
    train_parser = subparsers.add_parser('train', help='Train the model')

    # Define arguments for training
    train_parser.add_argument(
        "--input-data-path",
        type=str,
        default=os.path.join(script_path, "data", "raw"),
        help="Path to the input data directory (default: ./data/raw)",
    )
    train_parser.add_argument(
        "--working-data-path",
        type=str,
        default=os.path.join(script_path, "data", "processed"),
        help="Path to the working data directory (default: ./data/processed)",
    )
    train_parser.add_argument(
        "--results-path",
        type=str,
        default=os.path.join(script_path, "results"),
        help="Path to the results directory (default: ./results)",
    )
    train_parser.add_argument(
        "--checkpoints-path",
        type=str,
        default=os.path.join(script_path, "checkpoints"),
        help="Path to the checkpoints directory (default: ./checkpoints)",
    )

    # Argument for downloading dataset
    train_parser.add_argument(
        "--download-dataset",
        action="store_true",
        default=False,
        help="Download the dataset (default: False)",
    )

    # Subcommand for prediction
    predict_parser = subparsers.add_parser('predict', help='Make predictions on an image')

    # Define arguments for prediction
    predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file or URL for prediction",
    )

    # Subcommand for evaluation
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the models on test data')

    # Parse arguments
    args = parser.parse_args()

    # Handle the train command
    if args.command == 'train':
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
                print(f"Dataset already exists in {args.input_data_path}. Skipping download.")

        # Train the model
        train_model(args, num_epochs=10, seed=123, device=device)

    # Handle the predict command
    elif args.command == 'predict':
        CLASSES = ['drive', 'legglance-flick', 'pullshot', 'sweep']
        print(f"Making prediction on image: {args.image}")
        predict_shot(args.image, CLASSES, device)
    elif args.command == 'evaluate':
        print(f"Running evaluation on test data:")
        evaluate_models(script_path, device)


if __name__ == "__main__":
    main()
