# Prediksi Kelulusan Mahasiswa

Aplikasi Machine Learning untuk memprediksi kelulusan mahasiswa
menggunakan Decision Tree dan FastAPI.

## Fitur
- Training model Decision Tree
- Prediksi kelulusan via web
- Framework FastAPI

## Cara Menjalankan

### 1. Install dependency
python -m pip install -r requirements.txt

### 2. Training model
python train.py

### 3. Jalankan web
python -m uvicorn main:app --reload

Buka browser:
http://127.0.0.1:8000
