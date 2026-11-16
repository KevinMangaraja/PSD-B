# Business understanding
# Proyek Sains Data: Klasifikasi Citra Kerapian Sandal dan Sepatu (Alas Kaki)
## Dokumen Pemahaman Bisnis (CRISP-DM Tahap 1: Business Understanding)

---

### 1. Latar Belakang & Konteks Masalah
Penataan alas kaki (sandal dan sepatu) yang rapi merupakan indikator keteraturan, estetika, dan higienitas di berbagai ruang publik, institusi pendidikan, perkantoran, maupun tempat ibadah (masjid/musholla). Penumpukan alas kaki yang berserakan tidak hanya mengganggu kenyamanan dan mobilitas orang yang lewat, tetapi juga berpotensi menimbulkan kekacauan, kehilangan alas kaki, atau kesulitan proses evakuasi saat darurat.

Selama ini, proses pengawasan kerapian rak dan lantai alas kaki masih dilakukan secara manual oleh petugas kebersihan (*cleaning service*) atau pengelola fasilitas melalui patroli fisik berkala. Pendekatan ini memiliki kelemahan:
1. **Tidak Waktu-Nyata (*Real-Time*):** Ketidakteraturan sering kali terjadi secara masif pada jam-jam sibuk tanpa terdeteksi segera.
2. **Keterbatasan Sumber Daya Manusia:** Petugas tidak dapat memantau seluruh lorong dan rak alas kaki secara terus-menerus.
3. **Subjektivitas Penilaian:** Penilaian kerapian tanpa standar kuantitatif terukur.

Pemanfaatan teknologi **Computer Vision & Deep Learning (Convolutional Neural Networks / Vision Transformers)** memungkinkan sistem pemantauan cerdas berbasis kamera CCTV untuk secara otomatis mendeteksi dan mengklasifikasikan status kerapian alas kaki (*Tidy / Neat* vs *Untidy / Cluttered*) secara konsisten, objektif, dan otomatis.

---

### 2. Tujuan Bisnis (*Business Objectives*)
1. **Otomasi Pemantauan Kerapian:** Menghadirkan sistem visi komputer yang mampu menganalisis citra area rak/lantai alas kaki dan menentukan status kerapian secara instan.
2. **Efisiensi Operasional Pengelola Fasilitas:** Mengurangi frekuensi inspeksi manual dan memberikan notifikasi proaktif kepada petugas saat area alas kaki berantakan.
3. **Peningkatan Kualitas Layanan & Estetika Lingkungan:** Mempertahankan standar keteraturan fasilitas publik, masjid, sekolah, maupun gedung perkantoran.

---

### 3. Sasaran Proyek Sains Data (*Data Science Objectives*)
1. **Tugas Pemodelan (*Modeling Task*):** Membangun model klasifikasi citra untuk membedakan kategori:
   - **Rapi (*Tidy/Neat*):** Sandal/sepatu tertata sejajar, berada di dalam rak/kompartemen, atau berpasangan teratur.
   - **Tidak Rapi (*Untidy/Cluttered*):** Sandal/sepatu berserakan, bersilangan, bertumpuk, atau berada di luar area yang ditentukan.
   - **Baseline yang diimplementasikan:** Ekstraksi fitur statistik RGB/HSV, grayscale, tepi, kuadran, dan histogram gradien lokal (HOG), dilanjutkan perbandingan Extra Trees, Random Forest, Gradient Boosting, SVC, dan Logistic Regression. CNN/transfer learning tetap menjadi opsi pengembangan setelah data bertambah.
   *(Dapat dikembangkan menjadi multi-label/multi-class jika dataset mencakup segmentasi sandal vs sepatu).*
2. **Metrik Keberhasilan Teknis (*Technical Success Metrics*):**
   - **Akurasi Validasi & Pengujian:** $\ge 85,00\%$ pada *unseen test set*.
   - **Weighted F1-Score & Macro F1-Score:** $\ge 83,00\%$ untuk memitigasi potensi *class imbalance*.
   - **Inference Latency:** $< 100\text{ ms}$ per citra pada CPU/Edge Device untuk kelayakan deployment waktu-nyata.

---

### 4. Rencana Kerja Metodologi CRISP-DM

| Tahapan CRISP-DM | Rincian Aktivitas & Luaran | Direktori Kerja |
| :--- | :--- | :--- |
| **1. Business Understanding** | Perumusan masalah, batasan teknis, target metrik, dan arsitektur workflow. | `1-business_understanding/` |
| **2. Data Understanding** | Eksplorasi dataset citra, analisis resolusi, distribusi kelas, visualisasi sampel, dan identifikasi anomali. | `2-data_understanding/` |
| **3. Data Preparation** | Ekstraksi fitur visual dan split berdasarkan kelompok pengambilan (Kelompok 2 untuk test, sisanya untuk pengembangan) agar foto satu kelompok tidak bocor lintas split. | `3-data_preparation/` |
| **4. Modeling** | Perbandingan lima baseline machine learning klasik dengan StratifiedGroupKFold dan pemilihan berdasarkan rata-rata macro-F1; StandardScaler dipelajari per fold dan model terpilih dilatih ulang pada semua data pengembangan. | `4-modeling/` |
| **5. Evaluation** | Evaluasi pada kelompok test yang tidak digunakan saat pelatihan: accuracy, weighted/macro F1, classification report, confusion matrix, dan ROC-AUC. | `5-evaluation/` |
| **6. Deployment** | Pembuatan REST API inferensi menggunakan FastAPI, dashboard web unggah citra, kontainerisasi Docker & docker-compose. | `6-deployment/` |

---

### 5. Rencana Arsitektur Deployment
- **Backend Service:** FastAPI REST API dengan endpoint prediksi citra (`POST /api/v1/predict`) dan pemeriksaan status (`GET /health`).
- **Input Pipeline:** Pengunggahan citra (JPEG/PNG), pra-pemrosesan otomatis (resize ke tensor model, normalisasi $[0, 1]$ atau ImageNet standard), dan inferensi model.
- **Output:** Kelas prediksi (*Rapi* / *Tidak Rapi*), skor probabilitas keyakinan (*confidence score*), serta rekomendasi visual.

