import os
import subprocess
import zipfile
import shutil


def download_kaggle_dataset(dataset_name, download_path="."):
    """
    Download a Kaggle dataset using the Kaggle CLI and unzip it.

    Args:
        dataset_name (str): The name of the dataset to download (e.g., 'aneesh10/cricket-shot-dataset').
        download_path (str): The directory where the dataset will be downloaded.
    """
    # Check if Kaggle is configured
    kaggle_json_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            "Kaggle API credentials not found. Please ensure you've set up Kaggle CLI properly."
        )

    # Create download path if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Change the working directory to the download path
    os.chdir(download_path)

    # Download the dataset using Kaggle CLI
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name], check=True
        )
        print(f"Downloaded dataset: {dataset_name}")

        # Unzip the downloaded dataset
        for file in os.listdir(download_path):
            if file.endswith(".zip"):
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(download_path)
                print(f"Extracted: {file}")

                # After extraction, check if there is a 'data/' subfolder and move its contents up
                extracted_dir = os.path.join(download_path, "data")
                if os.path.exists(extracted_dir) and os.path.isdir(extracted_dir):
                    for item in os.listdir(extracted_dir):
                        shutil.move(os.path.join(extracted_dir, item), download_path)
                    os.rmdir(extracted_dir)  # Remove the now-empty 'data/' folder
                    print(f"Moved contents from 'data/' to {download_path}")

                # Optionally, delete the zip file after extraction
                os.remove(file)
                print(f"Deleted zip file: {file}")

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")


# Usage example
if __name__ == "__main__":
    download_kaggle_dataset("aneesh10/cricket-shot-dataset", download_path="./datasets")
