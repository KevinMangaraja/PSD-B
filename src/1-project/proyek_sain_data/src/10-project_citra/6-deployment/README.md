# Microservice Deployment: Klasifikasi Kerapian Sandal & Sepatu (Alas Kaki)

Layanan microservice REST API berbasis FastAPI untuk melakukan klasifikasi kerapian sandal dan sepatu pada rak atau lantai menggunakan model Computer Vision.

---

## 1. Menjalankan Layanan Secara Lokal

Jalankan perintah berikut menggunakan Python virtual environment:

```bash
cd "src/10-project_citra/6-deployment"
python -m uvicorn app:app --host 0.0.0.0 --port 8016 --reload
```

Akses layanan melalui browser:
* **Dashboard Unggah & Deteksi Citra:** [http://localhost:8016/](http://localhost:8016/)
* **Dokumentasi Interaktif API (Swagger UI):** [http://localhost:8016/docs](http://localhost:8016/docs)
* **Status Layanan (Health Check):** [http://localhost:8016/health](http://localhost:8016/health)

---

## 2. Menjalankan Layanan dengan Docker

### Opsi A: Menggunakan Docker CLI
```bash
cd "src/10-project_citra/6-deployment"
docker build -t footwear-classifier:latest .
docker run -d -p 8016:8016 --name footwear_app footwear-classifier:latest
```

### Opsi B: Menggunakan Docker Compose
```bash
cd "src/10-project_citra/6-deployment"
docker compose up -d --build
```

---

## 3. Spesifikasi Endpoint REST API

### `POST /api/v1/predict`
Mengunggah satu berkas citra untuk memperoleh hasil prediksi kerapian.

* **Content-Type:** `multipart/form-data`
* **Body:** `file` (binary image JPEG/PNG)
* **Response Contoh:**
```json
{
  "filename": "rak_sepatu_01.jpg",
  "image_size": [1920, 1080],
  "predicted_class": "Rapi (Tidy)",
  "confidence": 0.942,
  "class_probabilities": {
    "Rapi (Tidy)": 0.942,
    "Tidak Rapi (Untidy)": 0.058
  },
  "inference_latency_ms": 38.4,
  "status": "success"
}
```

