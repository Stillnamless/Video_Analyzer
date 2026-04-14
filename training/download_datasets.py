"""
Download script for training datasets.
- FER-2013 (from Kaggle) — for Emotion CNN
- RAVDESS  (from Zenodo) — for Voice Regressor

Requirements:
  pip install kaggle opendatasets requests

FER-2013 requires Kaggle credentials:
  1. Go to https://www.kaggle.com/settings and click "Create New API Token"
  2. Place the downloaded kaggle.json in ~/.kaggle/kaggle.json
"""
import os
import subprocess
import zipfile

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TRAINING_DIR, "data")
FER_DIR = os.path.join(DATA_DIR, "fer2013")
RAVDESS_DIR = os.path.join(DATA_DIR, "ravdess")

os.makedirs(FER_DIR, exist_ok=True)
os.makedirs(RAVDESS_DIR, exist_ok=True)


def download_fer2013():
    print("Downloading FER-2013 from Kaggle...")

    # Method 1: existing kaggle.json file
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "msambare/fer2013", "-p", FER_DIR, "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"FER-2013 downloaded to {FER_DIR}")
            return True
        print("Kaggle CLI failed:", result.stderr)

    # Method 2: interactive credentials via opendatasets
    print("\nNo kaggle.json found. Trying interactive login...")
    print("You'll need your Kaggle username + API key from https://www.kaggle.com/settings → 'Create New API Token'")
    try:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/msambare/fer2013", data_dir=DATA_DIR)
        # opendatasets saves to data_dir/fer2013/
        print(f"FER-2013 downloaded to {FER_DIR}")
        return True
    except ImportError:
        print("Installing opendatasets...")
        subprocess.run(["pip", "install", "opendatasets", "-q"])
        try:
            import opendatasets as od
            od.download("https://www.kaggle.com/datasets/msambare/fer2013", data_dir=DATA_DIR)
            return True
        except Exception as e:
            print(f"opendatasets failed: {e}")

    # Method 3: manual instructions
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Click 'Download' → Download ZIP")
    print(f"3. Extract to: {FER_DIR}")
    print("   It should contain: train/ and test/ subdirectories")
    print("="*60 + "\n")
    return False


def download_ravdess():
    """Download RAVDESS audio dataset (speech only, ~211MB)."""
    print("Downloading RAVDESS from Zenodo...")
    import urllib.request
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = os.path.join(RAVDESS_DIR, "ravdess.zip")

    if not os.path.exists(os.path.join(RAVDESS_DIR, "Actor_01")):
        print("Downloading... this may take a few minutes (~211 MB)")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAVDESS_DIR)
        os.remove(zip_path)
        print(f"RAVDESS downloaded to {RAVDESS_DIR}")
    else:
        print("RAVDESS already downloaded.")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("InterviewIQ Dataset Downloader")
    print("=" * 50)
    fer_ok   = download_fer2013()
    ravdess_ok = download_ravdess()
    print("\nSummary:")
    print(f"  FER-2013 : {'✓ Ready' if fer_ok else '✗ Failed (check Kaggle API key)'}")
    print(f"  RAVDESS  : {'✓ Ready' if ravdess_ok else '✗ Failed'}")
    if fer_ok and ravdess_ok:
        print("\nAll datasets ready! You can now run the training scripts.")
