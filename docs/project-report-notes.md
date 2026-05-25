# Report Notes

## Judul
Exam Readiness Predictor: Prediksi Kesiapan Ujian Menggunakan Backpropagation dan Adaline Berbasis Dataset Kaggle

## Dataset
Dataset utama: Student Habits vs Academic Performance dari Kaggle.

Slug Kaggle:

```txt
jayaantanaath/student-habits-vs-academic-performance
```

Aplikasi memakai `kagglehub` untuk mengunduh dataset. Jika internet tidak tersedia, file preview lokal digunakan agar UI tetap bisa didemokan.

## Model

1. Backpropagation ANN: prediksi exam_score.
2. Adaline: klasifikasi High Risk jika exam_score < 60.

## Package Tambahan

```txt
streamlit-option-menu
kagglehub
```
