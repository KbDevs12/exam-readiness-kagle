# Deployment Guide

## Local

```bash
pip install -r requirements.txt
python download_dataset.py
streamlit run app.py
```

## Streamlit Community Cloud

1. Push repository ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository dan file `app.py`.
4. Deploy.

Catatan: dataset Kaggle akan diunduh oleh `kagglehub` saat aplikasi dijalankan. Jika proses download gagal karena jaringan, app akan memakai preview CSV lokal dan menampilkan warning.
