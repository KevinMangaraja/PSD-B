# Proyek Sains Data — Portofolio & Implementasi CRISP-DM End-to-End

Repositori ini memuat kumpulan proyek dan penelitian **Proyek Sains Data (*Data Science*)** yang disusun menggunakan metodologi standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)**, mulai dari *Business Understanding*, *Data Understanding*, *Data Preparation*, *Modeling*, *Evaluation*, hingga *Deployment*. Seluruh dokumentasi juga terintegrasi dengan **MyST Markdown (`mystmd`)** sebagai buku interaktif web.

---

## Profil Penulis

| Atribut | Keterangan |
| :--- | :--- |
| **Nama Lengkap** | **Muhammad Ilham** |
| **NIM** | `240411100187` |
| **Program Studi / Jurusan** | Teknik Informatika |
| **Institusi** | Universitas Trunojoyo Madura (UTM) |
| **Email** | ilhamm8615@gmail.com |
| **GitHub** | [github.com/IlhamInsight](https://github.com/IlhamInsight) |

---

## Daftar Proyek dalam Repositori

1. **Tugas 1 — Eksplorasi & Persiapan Data Awal (`src/1-bussines_understanding` s.d. `src/4-data_preparation`)**
   * Pemahaman proses bisnis, teknik pengumpulan/pengambilan data, eksplorasi distribusi statistik (*Data Understanding*), dan pembersihan data (*Data Preparation*).
2. **Tugas 2 — Pipeline Pemodelan berbasis Data Crawling (`src/tugas-2`)**
   * Akuisisi data melalui *Web Crawling*, eksplorasi karakteristik fitur, *preprocessing*, pemodelan *machine learning*, dan evaluasi kinerja model.
3. **Tugas UTS — Analisis & Pemodelan Prediktif End-to-End (`src/Tugas_UTS`)**
   * Implementasi siklus penuh 6 tahap CRISP-DM beserta aplikasi *deployment* interaktif.
4. **Project Akhir — Speech Emotion Recognition (SER) Berbasis Attention-based CRNN pada Dataset RAVDESS (`src/project_akhir`)**
   * Sistem pengenalan emosi manusia dari sinyal suara vokal (*Audio/Speech*) untuk **8 kelas emosi** menggunakan arsitektur tunggal **Attention-based CRNN (2D-CNN + Bidirectional LSTM + Multi-Head Self-Attention)**, dilengkapi strategi **Hybrid Class Balancing**, **Speaker-Independent Validation**, serta **Deployment Real-Time (`FP16 Runtime`)** dengan kalibrasi lintas bahasa dan logat daerah.

---

## Ringkasan Teknis Project Akhir: Speech Emotion Recognition (SER)

### 1. Daftar 8 Kelas Emosi Vokal (Dataset RAVDESS)

Dataset menggunakan **RAVDESS (*Ryerson Audio-Visual Database of Emotional Speech and Song*)** porsi *Audio-Only Speech* (`1.440 file .wav`, `24 aktor` — 12 Pria & 12 Wanita).

| Kode | Nama Kelas (Inggris) | Terjemahan (Indonesia) | Sampel Asli | Sampel Train Asli | Sampel Train Pasca-Balancing | Karakteristik Psikologis (*Arousal & Valence*) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **`01`** | **`Neutral`** | Netral / Biasa | `96` *(Minoritas)* | `72` | **`288`** *(Augmentasi $4\times$)* | *Low–Medium Arousal*, *Neutral Valence* |
| **`02`** | **`Calm`** | Tenang / Rileks | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *Low Arousal*, *Positive Valence* |
| **`03`** | **`Happy`** | Bahagia / Senang | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *High Arousal*, *Positive Valence* |
| **`04`** | **`Sad`** | Sedih / Muram | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *Low Arousal*, *Negative Valence* |
| **`05`** | **`Angry`** | Marah / Tegang | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *Very High Arousal*, *Negative Valence* |
| **`06`** | **`Fearful`** | Takut / Cemas | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *High Arousal*, *Negative Valence* |
| **`07`** | **`Disgust`** | Jijik / Tidak Suka | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *Medium Arousal*, *Negative Valence* |
| **`08`** | **`Surprised`** | Terkejut / Kaget | `192` | `144` | **`288`** *(Augmentasi $2\times$)* | *High Arousal*, *Positive–Neutral Valence* |

### 2. Metodologi & Arsitektur (`Attention-based CRNN`)

* **Pra-pemrosesan Sinyal & Ekstraksi Fitur Multi-Domain (`264 Kanal × 94 Frame`):**
  * Konversi *Mono*, *Resampling* ke `16.000 Hz`, *Silence Trimming (`25 dB`)*, *Pre-Emphasis Filter ($\alpha = 0,97$)*, dan penyeragaman durasi `3,0 detik` (`48.000 sampel`).
  * Ekstraksi **Log Mel-Spectrogram (`128 bin`)**, **MFCC (`40 Statis + 40 Delta + 40 Delta-Delta = 120 kanal`)**, **Chromagram (`12 pitch class`)**, serta **Fitur Prosodi & Spektral (`4 kanal`: *RMS Energy*, *ZCR*, *Spectral Centroid*, *Spectral Rolloff*)**.
* **Pembagian Data *Speaker-Independent* & *Hybrid Class Balancing*:**
  * **Training Set (`Actor 01–18`):** `1.080` sampel asli $\to$ diseimbangkan melalui *Targeted Audio Augmentation Oversampling* menjadi **`2.304` sampel (`288 sampel/kelas`)**.
  * **Validation Set (`Actor 19–20`):** `120` sampel asli murni.
  * **Testing Set (`Actor 21–24`):** `240` sampel asli murni (*Unseen Speakers*).
  * **Level Algoritma:** Menggunakan *Inverse Class-Frequency Weighting ($w_k$)* pada *Weighted Categorical Cross-Entropy Loss*.
* **Hasil Evaluasi Model:**
  * **Testing Set (`Actor 21–24`, Penutur Baru):** `WAR / Accuracy = 90,42%` | `UAR / Macro Recall = 91,02%` | `Macro Precision = 90,62%` | `Macro F1 = 90,64%` | `ROC-AUC = 99,31%`.
  * **5-Fold Speaker-Independent Cross-Validation (`24 Aktor`):** `WAR = 95,14%` | `UAR = 95,44%` | `Macro Precision = 94,76%` | `Cohen's Kappa = 0,9442`.
  * **Deployment Runtime (`FP16`):** Ukuran memori bobot berkurang `50%` (`697,27 KB`) dengan latensi inferensi model `< 1,0 ms` (target bisnis `< 40 ms` terpenuhi).

---

## Struktur Direktori Repositori

```text
proyek sain data/
├── .streamlit/
│   └── config.toml                                    # Konfigurasi tema gelap (#000000) untuk Streamlit
├── src/
│   ├── 0-personal_data/
│   │   └── 1-personal_data.md                         # Profil penulis pada buku MyST
│   ├── 1-bussines_understanding/                      # Tugas 1: Business Understanding
│   ├── 2-pengambilan_data/                            # Tugas 1: Pengambilan Data
│   ├── 3-data_understanding/                          # Tugas 1: Data Understanding
│   ├── 4-data_preparation/                            # Tugas 1: Data Preparation
│   ├── tugas-2/                                       # Tugas 2: Pipeline Crawling s.d. Evaluation
│   ├── Tugas_UTS/                                     # Tugas UTS: Pipeline CRISP-DM 6 Tahap
│   └── project_akhir/                                 # PROJECT AKHIR: Speech Emotion Recognition (RAVDESS)
│       ├── 1-business_understading/
│       │   ├── 1-business.md                          # Dokumen spesifikasi bisnis, matematis, & arsitektur CRNN
│       │   └── ser_pipeline_flowchart.svg             # Diagram alir vektor arsitektur CRISP-DM SER
│       ├── 2-data_understanding/
│       │   ├── 1-data_understanding.ipynb             # Notebook audit sinyal, distribusi kelas, & EDA akustik
│       │   ├── ravdess_data_understanding_catalog.csv # Katalog metadata & fitur dasar 1.440 file audio
│       │   ├── dataset/                               # [Diabaikan di Git] 1.440 file .wav RAVDESS
│       │   └── images/                                # Visualisasi waveform, spectrogram, MFCC, & boxplot
│       ├── 3-data_preparation/
│       │   ├── 1-data_preparation.ipynb               # Notebook trimming, augmentasi, ekstraksi 264 kanal, & Z-Score
│       │   ├── data/                                  # Manifest data seimbang (2.304 train) & metadata partisi
│       │   └── images/                                # Visualisasi tahapan pembersihan sinyal & class balancing
│       ├── 4-modelling/
│       │   ├── 1-modeling.ipynb                       # Notebook pelatihan Attention-based CRNN & 30-Trial Optuna TPE
│       │   ├── models/                                # Riwayat training, hasil 5-Fold CV, & log trial Optuna
│       │   └── images/                                # Kurva loss/UAR, optimasi Optuna, & peta bobot Self-Attention
│       ├── 5-evaluation/
│       │   ├── 1-evaluation.ipynb                     # Notebook evaluasi metrik per-kelas, WAR/UAR, & keadilan gender
│       │   ├── results/                               # Tabel CSV hasil evaluasi Test Set & 5-Fold CV
│       │   └── images/                                # Confusion matrix 8x8, kurva ROC/PR, & analisis sub-kelompok
│       └── 6-deploy/
│           ├── 1-deployment.ipynb                     # Notebook eksportasi FP16, benchmark latensi, & pemetaan afektif
│           ├── app.py                                 # Aplikasi Web Interaktif Streamlit (Kalibrasi Lintas Bahasa/Logat)
│           ├── api_server.py                          # Layanan Mikro REST API (Starlette / Uvicorn)
│           ├── requirements.txt                       # Dependensi paket deployment
│           ├── models/
│           │   ├── attention_crnn_fp16_runtime.npz    # Bobot model runtime FP16 & dekoder akustik bebas gain
│           │   └── deployment_test_inference_log.csv  # Log hasil inferensi deployment pada 240 sampel uji
│           └── images/                                # Grafik benchmark FP16, preview dashboard, & Russell Circumplex
├── .gitignore                                         # Aturan pengecualian dataset (.wav), venv, dan cache
├── myst.yml                                           # Konfigurasi daftar isi (TOC) buku web MyST Markdown
├── requirements.txt                                   # Daftar pustaka Python utama
└── README.md                                          # Dokumentasi utama repositori
```

> **Catatan Dataset pada Git (`.gitignore`):**
> Folder dataset mentah `src/project_akhir/2-data_understanding/dataset/` (`1.440` file `.wav` berukuran $\approx 1,2\text{ GB}$) serta file binari perantara sengaja dikecualikan di dalam `.gitignore` agar repositori tetap ringan dan cepat saat di-*push* ke GitHub. Artefak ringkas hasil ekstraksi dan bobot runtime `FP16` (`attention_crnn_fp16_runtime.npz`) tetap disertakan agar aplikasi *deployment* dapat langsung dijalankan.

---

## Cara Menjalankan Proyek

### 1. Persiapan Lingkungan Virtual (*Virtual Environment*)

```powershell
# Buat atau aktifkan virtual environment
.\venv\Scripts\activate

# Instal dependensi yang diperlukan
.\venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Menjalankan Aplikasi Web Deployment Interaktif (Streamlit — `Project Akhir`)

Cukup jalankan **satu perintah** berikut dari direktori utama proyek:

```powershell
.\venv\Scripts\streamlit.exe run "src/project_akhir/6-deploy/app.py"
```

* Buka di browser: **http://localhost:8501**
* **Fitur Utama Aplikasi Web:**
  1. **3 Mode Input Suara:** Pilih sampel dari katalog RAVDESS, unggah file `.wav` eksternal, atau rekam suara langsung melalui mikrofon (*Live Mic*) tanpa perlu me-*refresh* halaman browser saat berganti input.
  2. **Kalibrasi Lintas Bahasa & Logat Daerah (*Utterance-CMVN & Kontur Pitch $F_0$*):** Dilengkapi pilihan profil adaptasi prosodi untuk Bahasa Indonesia dan logat daerah (logat intonasi tegas/energi tinggi, logat halus/tempo lambat, logat cepat/melodis) agar klasifikasi kelas peringkat #1 akurat dan tidak bias terhadap *gain* mikrofon.
  3. **Rincian 8 Kelas Emosi (dengan Terjemahan Indonesia):** Menampilkan kelas terdeteksi, persentase akurasi klasifikasi sampel (`85% – 95%`), selisih margin terhadap kandidat ke-2, tabel perbandingan seluruh 8 kelas, indikator pembeda akustik, serta visualisasi *Waveform + Self-Attention*, *Log Mel-Spectrogram*, dan *MFCC*.

### 3. Menjalankan Layanan Mikro REST API (Opsional)

```powershell
.\venv\Scripts\uvicorn.exe api_server:app --app-dir "src/project_akhir/6-deploy" --host 127.0.0.1 --port 8000
```

* Endpoint status kesehatan: `GET http://127.0.0.1:8000/api/v1/health`
* Endpoint prediksi file `.wav`: `POST http://127.0.0.1:8000/api/v1/predict-emotion`

### 4. Menjalankan Dokumentasi Buku Interaktif MyST Markdown

```powershell
npx mystmd start
```

