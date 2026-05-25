# Exam Readiness Predictor - Kaggle Dataset

Aplikasi Streamlit untuk memprediksi kesiapan ujian mahasiswa menggunakan dataset Kaggle **Student Habits vs Academic Performance**.

Model JST yang digunakan dari nol:

1. **Backpropagation ANN** untuk memprediksi nilai ujian (`exam_score`).
2. **Adaline** untuk klasifikasi risiko performa rendah (`High Risk` / `Low Risk`).

Dataset utama diunduh otomatis dari Kaggle menggunakan `kagglehub`.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Download dataset Kaggle secara manual

```bash
python download_dataset.py
```

Jika komputer belum terhubung internet, aplikasi tetap bisa dibuka memakai file preview `data/sample_student_habits_preview.csv`. Untuk laporan final, gunakan dataset Kaggle asli dengan menjalankan `python download_dataset.py`.


## Dataset

Dataset utama: **Student Habits vs Academic Performance** dari Kaggle.

Slug:

```txt
jayaantanaath/student-habits-vs-academic-performance
```

Aplikasi memanggil `kagglehub.dataset_download()` untuk mengambil dataset.

## Struktur Project

```txt
exam-readiness-kaggle/
├── app.py
├── data_loader.py
├── preprocessing.py
├── pipeline.py
├── recommendation.py
├── training_utils.py
├── download_dataset.py
├── requirements.txt
├── data/
│   └── sample_student_habits_preview.csv
├── models/
│   ├── __init__.py
│   ├── backprop_ann.py
│   └── adaline.py
└── docs/
    ├── commit-plan.md
    ├── deployment-guide.md
    └── project-report-notes.md
```

## Push ke GitHub

```bash
git remote add origin https://github.com/USERNAME/exam-readiness-kaggle.git
git branch -M main
git push -u origin main
```
