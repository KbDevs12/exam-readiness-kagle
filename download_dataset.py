from pathlib import Path
import shutil

KAGGLE_DATASET = "jayaantanaath/student-habits-vs-academic-performance"
TARGET_PATH = Path("data/student_habits_kaggle.csv")


def main():
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Package kagglehub belum terinstall. Jalankan: pip install kagglehub") from exc

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = list(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise SystemExit("Tidak ada file CSV yang ditemukan dari dataset Kaggle.")

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_files[0], TARGET_PATH)
    print(f"Dataset disimpan ke: {TARGET_PATH}")
    print("Selesai. Jalankan: streamlit run app.py")


if __name__ == "__main__":
    main()
