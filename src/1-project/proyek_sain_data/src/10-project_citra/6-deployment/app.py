import io
import time
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import joblib
from feature_extraction import FEATURE_COLUMNS, extract_image_features

app = FastAPI(
    title="Footwear Tidiness Classifier API",
    description="API Klasifikasi Kerapian Sandal dan Sepatu (Alas Kaki) Berbasis Computer Vision & Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["Tidak Rapi (Berantakan)", "Rapi (Tidy)"]

# Load Model, Scaler, and Features configuration
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "best_model.joblib"
SCALER_PATH = CURRENT_DIR / "scaler.joblib"
FEATURE_COLS_PATH = CURRENT_DIR / "feature_columns.json"
if not FEATURE_COLS_PATH.exists():
    FEATURE_COLS_PATH = CURRENT_DIR.parent / "4-modeling" / "models" / "feature_columns.json"

model = None
scaler = None
feature_cols = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH, "r") as f:
            feature_cols = json.load(f)
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    print("Model, scaler, and feature definitions loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load model/scaler: {e}")

def extract_features_from_image(img_rgb):
    features = extract_image_features(img_rgb)
    columns = feature_cols or FEATURE_COLUMNS
    try:
        return [features[column] for column in columns]
    except KeyError as e:
        raise RuntimeError(f"Fitur model tidak dikenali: {e.args[0]}") from e

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistem Pemantau Kerapian Alas Kaki</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }
            .container { max-width: 650px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; }
            h1 { font-size: 1.5rem; margin-bottom: 0.5rem; color: #38bdf8; }
            p { color: #94a3b8; font-size: 0.95rem; }
            .dropzone { border: 2px dashed #475569; padding: 2rem; text-align: center; border-radius: 8px; margin: 1.5rem 0; cursor: pointer; }
            .btn { background: #0284c7; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-weight: 600; cursor: pointer; }
            .btn:hover { background: #0369a1; }
            #preview { max-width: 100%; max-height: 250px; margin-top: 1rem; border-radius: 6px; display: none; }
            #result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; background: #0f172a; display: none; border: 1px solid #334155; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deteksi Kerapian Sandal & Sepatu</h1>
            <p>Unggah foto rak atau lantai alas kaki untuk memverifikasi tingkat kerapian secara otomatis.</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <div class="dropzone" onclick="document.getElementById('imageInput').click()">
                Klik di sini untuk memilih gambar sandal/sepatu
            </div>
            <center><img id="preview"></center>
            <center><button class="btn" id="analyzeBtn" onclick="analyzeImage()" style="display: none;">Analisis Kerapian</button></center>
            <div id="result"></div>
        </div>
        <script>
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const resultDiv = document.getElementById('result');

            input.onchange = e => {
                const [file] = input.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = 'block';
                    analyzeBtn.style.display = 'inline-block';
                    resultDiv.style.display = 'none';
                }
            };

            async function analyzeImage() {
                const file = input.files[0];
                if (!file) return;
                const formData = new FormData();
                formData.append('file', file);
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Memproses analisis visi komputer...';

                try {
                    const res = await fetch('/api/v1/predict', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        const statusColor = data.predicted_class.includes('Rapi') && !data.predicted_class.includes('Tidak') ? '#4ade80' : '#f87171';
                        resultDiv.innerHTML = `
                            <h3 style="color: ${statusColor}; margin-top: 0;">Hasil: ${data.predicted_class}</h3>
                            <p><strong>Tingkat Keyakinan:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                            <p><strong>Probabilitas Rapi:</strong> ${(data.class_probabilities['Rapi (Tidy)'] * 100).toFixed(2)}%</p>
                            <p><strong>Probabilitas Berantakan:</strong> ${(data.class_probabilities['Tidak Rapi (Berantakan)'] * 100).toFixed(2)}%</p>
                            <p><strong>Waktu Inferensi:</strong> ${data.inference_latency_ms.toFixed(1)} ms</p>
                        `;
                    } else {
                        resultDiv.innerHTML = `<span style="color: #ef4444;">Error: ${data.detail || 'Gagal memproses gambar'}</span>`;
                    }
                } catch(err) {
                    resultDiv.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    model_ready = model is not None and scaler is not None
    return {
        "status": "healthy" if model_ready else "unavailable",
        "service": "footwear-tidiness-classifier",
        "model_loaded": model_ready,
        "timestamp": time.time()
    }

@app.post("/api/v1/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berformat gambar.")
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses file citra: {str(e)}")
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau Scaler belum dimuat di server.")
    
    # Ekstraksi fitur visual
    feats = extract_features_from_image(image)
    X = np.array(feats).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    pred_label = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]
    
    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probs)
    }
    if 0 not in class_probabilities or 1 not in class_probabilities:
        raise HTTPException(status_code=500, detail="Model tidak memiliki dua kelas yang diharapkan.")

    prob_berantakan = class_probabilities[0]
    prob_rapi = class_probabilities[1]
    
    confidence = float(max(prob_rapi, prob_berantakan))
    predicted_class = "Rapi (Tidy)" if pred_label == 1 else "Tidak Rapi (Berantakan)"
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "filename": file.filename,
        "image_size": image.size,
        "predicted_class": predicted_class,
        "label": pred_label,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            "Rapi (Tidy)": round(prob_rapi, 4),
            "Tidak Rapi (Berantakan)": round(prob_berantakan, 4)
        },
        "inference_latency_ms": round(latency_ms, 2),
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8016, reload=True)
