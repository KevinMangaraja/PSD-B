# Data Preparation — Klasifikasi Land Use and Land Cover (LULC) Jawa Timur

Dokumen ini menjelaskan tahap **Data Preparation** dalam metodologi CRISP-DM untuk proyek UTS LULC Jawa Timur. Tahap ini mencakup ekstraksi fitur spektral, kalkulasi indeks spektral, pembersihan data, serta pembagian dataset (_train-test split_) berbasis polygon ID untuk menghindari _data leakage_.

---

## 3. Data Preparation

### 3.1 Ekstraksi Nilai Spektral & Rekayasa Fitur (Feature Engineering)

Data masukan (_input features_) diambil dari piksel citra satelit Sentinel-2 yang jatuh di dalam poligon sampel.
Maksimal 50 piksel valid dipilih secara deterministik untuk setiap poligon agar ukuran dataset tetap terkendali dan konsisten dengan format CSV terdahulu.
Pembacaan raster JP2 pada notebook memerlukan paket Python `rasterio`.

#### Fitur Spektral yang Diekstrak:

1. **Raw Bands (5 Fitur):**
   - `B02` (Blue)
   - `B03` (Green)
   - `B04` (Red)
   - `B08` (Near-Infrared / NIR)
   - `B11` (Shortwave-Infrared / SWIR1)

2. **Indeks Spektral Kunci (3 Fitur Turunan):**
   - `NDVI` $= \frac{B08 - B04}{B08 + B04}$ (Indeks Kerapatan Vegetasi)
   - `NDWI` $= \frac{B03 - B08}{B03 + B08}$ (Indeks Kebasahan Air)
   - `NDBI` $= \frac{B11 - B08}{B11 + B08}$ (Indeks Area Terbangun / Pemukiman)

---

### 3.2 Pembagian Data Train-Test Berbasis Polygon (Polygon-Level Split)

Untuk mencegah kebocoran data (_spatial autocorrelation data leakage_), pembagian data latih (70%) dan data uji (30%) dilakukan di tingkat **Polygon ID**, bukan di tingkat piksel individu.

- **Aturan Split:** Stratified split berbasis `ID_Poligon` dan kelas `Type`, dengan alokasi sisa 30% deterministik (sisa pecahan terbesar; seri dipecahkan menurut nama kelas).
- **Proporsi Split:**
  - **Alokasi awal:** 193 poligon latih dan 83 poligon uji dari 276 poligon acuan.
  - Poligon tanpa piksel Sentinel-2 valid dikeluarkan setelah ekstraksi, kemudian split 70:30 dihitung ulang per kelas untuk poligon yang tersisa; jumlah akhir dicatat pada output notebook.
  - Pembagian per kelas mengikuti jumlah sampel yang tidak seimbang: `Sawah` 37/15, `Air` 39/17, `Pemukiman` 43/19, `Hutan mangrove` 29/13, dan `Hutan non mangrove` 45/19 (latih/uji).

> Piksel dari polygon yang sama **tidak pernah** dipisah ke dalam data latih dan data uji secara bersamaan. Hal ini menjamin evaluasi akurasi model benar-benar objektif pada wilayah yang belum pernah dilihat model (_unseen spatial area_).

GeoJSON baru hanya berisi poligon dan label, bukan nilai band Sentinel-2. Dua fitur degeneratif dihapus (geometri sumber di `data/legacy/invalid_geometry_features.geojson`) dan lima geometri self-intersect diperbaiki dengan `make_valid` (geometri sumber di `data/legacy/repaired_geometry_sources.geojson`). CSV piksel lama (8.750 latih dan 3.750 uji) berasal dari 250 poligon sebelumnya dan telah dihapus agar tidak tertukar dengan hasil terbaru. Model lama tetap berada di `data/legacy/model_rf_lulc_250_polygons.joblib` sebagai arsip, tetapi tidak digunakan. Notebook mengekstrak band dari `data/geojson/TILE_1` sampai `TILE_5`, mengabaikan piksel bernilai nol, dan mengecualikan poligon tanpa piksel citra valid. Karena band SCL tidak disediakan, masking awan tidak dilakukan. Model terbaru disimpan ke `data/model_rf_lulc.joblib`; pelatihan, evaluasi, dan feature importance dijalankan setelah ekstraksi berhasil.

---

### 3.3 Ekspor & Struktur Storage CSV

Ringkasan poligon terbaru diekspor ke dalam folder `data/csv/split_dataset/`:

1. `train_polygons.csv` dan `test_polygons.csv` — ID serta kelas poligon yang memiliki piksel citra valid.
2. `train_pixels.csv` dan `test_pixels.csv` — Maksimal 50 sampel piksel valid per poligon, masing-masing dengan `ID_Poligon`, `Type`, `Status_Split`, lima band mentah, dan tiga indeks spektral.
