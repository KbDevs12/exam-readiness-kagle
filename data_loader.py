from pathlib import Path
import shutil
import pandas as pd

KAGGLE_DATASET = "jayaantanaath/student-habits-vs-academic-performance"
LOCAL_KAGGLE_CSV = Path("data/student_habits_kaggle.csv")
PREVIEW_CSV = Path("data/sample_student_habits_preview.csv")


def download_from_kaggle():
    """Download dataset Kaggle dan simpan ke folder data/."""
    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = list(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Dataset Kaggle berhasil diakses, tetapi CSV tidak ditemukan.")

    LOCAL_KAGGLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_files[0], LOCAL_KAGGLE_CSV)
    return LOCAL_KAGGLE_CSV


def load_raw_dataset(allow_preview=True):
    """
    Prioritas data:
    1. data/student_habits_kaggle.csv jika sudah ada.
    2. Download otomatis dari Kaggle via kagglehub.
    3. Preview CSV lokal jika internet tidak tersedia.
    """
    if LOCAL_KAGGLE_CSV.exists():
        return pd.read_csv(LOCAL_KAGGLE_CSV), "Kaggle CSV lokal: data/student_habits_kaggle.csv", False

    try:
        csv_path = download_from_kaggle()
        return pd.read_csv(csv_path), "Dataset Kaggle diunduh otomatis via kagglehub", False
    except Exception as exc:
        if allow_preview and PREVIEW_CSV.exists():
            return pd.read_csv(PREVIEW_CSV), f"Preview lokal digunakan karena Kaggle belum bisa diakses: {exc}", True
        raise
