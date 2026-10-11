# Data Understanding — Klasifikasi Land Use and Land Cover (LULC) Jawa Timur

Dokumen ini menjelaskan tahap **Data Understanding** dalam metodologi CRISP-DM untuk proyek UTS LULC Jawa Timur. Tahap ini berfokus pada analisis sampel data vektor serta pemahaman mendalam mengenai karakteristik dan alasan penggunaan band spektral citra satelit Sentinel-2.

---

## 2. Data Understanding

### 2.1 Audit & Karakteristik Data Vektor (GeoJSON)

Data acuan (_ground truth_) dibaca dari `data/geojson/jawa_timur.geojson`. Dua fitur dengan geometri degeneratif dihapus dan disimpan di `data/legacy/invalid_geometry_features.geojson`; lima geometri self-intersect diperbaiki menjadi geometri valid dan geometri aslinya disimpan di `data/legacy/repaired_geometry_sources.geojson`. Berkas aktif berisi 276 poligon valid dengan ID stabil `ID_Poligon` dan atribut kelas distandardisasi menjadi kolom `Type`. Nilai label yang memiliki spasi/baris baru atau label `hutan non mangrovehutan non mangrove` yang tergabung dikoreksi menjadi nama kelas kanonis.

| Kelas | Jumlah poligon |
| :--- | ---: |
| Sawah | 52 |
| Air | 56 |
| Pemukiman | 62 |
| Hutan mangrove | 42 |
| Hutan non mangrove | 64 |
| **Total** | **276** |

Jumlah sampel tidak seimbang. Geometri yang tersisa dapat dibaca sebagai poligon; tumpang-tindih spasial dan kesesuaian label tetap perlu diperiksa.

GeoJSON hanya menyediakan geometri dan label kelas, bukan nilai spektral Sentinel-2. Nilai band kini diekstrak dari raster JP2 di folder `TILE_1` sampai `TILE_5`; band B11 20 m di-resample bilinear ke grid 10 m. Piksel bernilai nol dibuang, dan poligon tanpa piksel citra valid tidak masuk ke split akhir. Maksimal 50 piksel dipilih secara deterministik per poligon. Cloud masking tidak dilakukan karena band SCL tidak disertakan. CSV piksel lama dari dataset 250 poligon telah dihapus; model lama tetap diarsipkan dan tidak digunakan.

---

### 2.2 Band Satelit Sentinel-2 yang Dibutuhkan & Alasan Penggunaannya

Untuk mengklasifikasikan 5 kelas tutupan lahan secara presisi, dibutuhkan **5 Band Spektral Kunci** dari sensor optik Sentinel-2 (Level-2A):

| Band Satelit      | Nama Gelombang             | Panjang Gelombang ($\lambda$) | Resolusi Spasial | Alasan Mengapa Band Ini Dibutuhkan                                                                                                                                                                                                                        |
| :---------------- | :------------------------- | :---------------------------: | :--------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Band 2 (B02)**  | Blue (Biru)                |            ~490 nm            |     10 meter     | **Penetrasi Air & Komposit Warna:** Cahaya biru dapat menembus kolom air dan sangat baik membedakan fitur perairan dangkal/sedimentasi serta digunakan untuk komposit warna alami (RGB).                                                                  |
| **Band 3 (B03)**  | Green (Hijau)              |            ~560 nm            |     10 meter     | **Deteksi Kebasahan & Air (NDWI):** Vegetasi dan air memantulkan cahaya hijau. Band ini dibutuhkan sebagai komponen utama formula **NDWI** untuk memisahkan badan air dari daratan.                                                                       |
| **Band 4 (B04)**  | Red (Merah)                |            ~665 nm            |     10 meter     | **Penyerapan Klorofil (NDVI):** Klorofil tanaman menyerap kuat spektrum merah. Kombinasi Red dan NIR dibutuhkan untuk menghitung **NDVI** guna mengukur kelebatan vegetasi.                                                                               |
| **Band 8 (B08)**  | NIR (Near-Infrared)        |            ~842 nm            |     10 meter     | **Diferensiasi Vegetasi vs Non-Vegetasi:** Struktur sel mesofil vegetasi sehat memantulkan NIR sangat tinggi, sedangkan air menyerap NIR sepenuhnya. Kunci memisahkan hutan dari pemukiman/air.                                                           |
| **Band 11 (B11)** | SWIR1 (Shortwave-Infrared) |           ~1610 nm            |     20 meter     | **Diferensiasi Mangrove & Pemukiman (NDBI):** SWIR sangat sensitif terhadap kelembapan tanah/daun dan material bangunan (beton/atap). Kunci utama membedakan **Hutan Mangrove** (basah) vs **Hutan Darat** (kering) serta mengidentifikasi **Pemukiman**. |

---

### 2.3 Mengapa Kombinasi Band Ini Sangat Krusial?

1. **Membedakan Hutan Mangrove vs Hutan Non-Mangrove:**
   Secara visual (RGB/Mata telanjang), kedua jenis hutan ini sama-sama berwarna hijau lebat. Namun pada **Band 11 (SWIR)**, Hutan Mangrove menunjukkan pantulan yang jauh lebih rendah dibanding Hutan Daratan karena substrat/tanah di bawah kanopi mangrove selalu basah atau tergenang air laut.

2. **Membedakan Pemukiman (Built-Up) vs Tanah Olahan Sawah:**
   Material buatan manusia (beton, semen, genteng) pada **Pemukiman** memiliki nilai pantulan **SWIR (B11)** yang sangat tinggi disertai nilai **NIR (B08)** sedang, sehingga menghasilkan nilai indeks **NDBI** bernilai positif.

3. **Membedakan Badan Air vs Kelas Lainnya:**
   Air menyerap hampir seluruh energi pada spektrum **NIR (B08)** dan **SWIR (B11)**, menjadikannya tampak sangat gelap (nilai mendekati 0), sementara pada **Band 3 (Green)** memiliki pantulan relatif tinggi. Hal ini membuat indeks **NDWI** sangat efektif memisahkan air secara mutlak.

---

### 2.4 Indeks Spektral Tambahan (Feature Engineering)

Dari 5 band di atas, dihitung 3 Indeks Spektral untuk memperkuat performa algoritma Machine Learning:

$$\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}} \quad \text{(Indeks Kerapatan Vegetasi)}$$

$$\text{NDWI} = \frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}} \quad \text{(Indeks Badan Air / Kebasahan)}$$

$$\text{NDBI} = \frac{\text{B11} - \text{B08}}{\text{B11} + \text{B08}} \quad \text{(Indeks Bangunan / Pemukiman)}$$
