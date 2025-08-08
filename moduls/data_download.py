import zipfile
from pathlib import Path
import gdown
import pandas as pd


def data_download(file_id: str) -> Path:
    """
    Downloads and extracts the malicious_url dataset if it does not already exist.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    dataset_path = data_path / "db"

    data_path.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    print(f"Creating dataset directory at {dataset_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_path / "malicious_url.zip"
    print("Downloading dataset ...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False
    )

    print("Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    zip_path.unlink()
    print(f"Dataset ready at {dataset_path}")
    return dataset_path


def data_info(path: Path) -> None:
    """
    Displays basic information about the malicious URL dataset.
    """
    csv_path = path / "malicious_phish1.csv"
    data = pd.read_csv(csv_path)

    print("\n Dataset Info:")
    data.info()
    print("\n URL types:", data["type"].unique())
    print("\n URL type distribution:")
    print(data["type"].value_counts())


if __name__ == "__main__":
    dataset_path = data_download("16rCT2HMQa6lnQfLYoGGC94aRMBj1O9r9")
    data_info(dataset_path)
