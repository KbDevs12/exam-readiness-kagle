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
