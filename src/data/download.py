from pathlib import Path
import subprocess
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

COMPETITION = "home-credit-default-risk"
ZIP_FILE = RAW_DIR / f"{COMPETITION}.zip"


def download_dataset() -> None:
    """Download and extract the Home Credit dataset from Kaggle."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Home Credit Default Risk dataset...")

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(RAW_DIR),
        ],
        check=True,
    )

    print("Download complete.")

    if ZIP_FILE.exists():
        print("Extracting dataset...")

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)

        print("Extraction complete.")
        ZIP_FILE.unlink()
    else:
        print("Dataset ZIP file was not found.")


if __name__ == "__main__":
    download_dataset()