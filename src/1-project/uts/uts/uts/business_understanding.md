# Business Understanding — Klasifikasi Land Use and Land Cover (LULC) Jawa Timur

Dokumen ini menjelaskan secara mendalam tahap **Business Understanding** dalam metodologi CRISP-DM untuk proyek Tugas Pengganti UTS: **Klasifikasi Land Use and Land Cover (LULC) Provinsi Jawa Timur** berbasis Citra Satelit Optik Sentinel-2.

---

## 1. Business Understanding

### 1.1 Apakah itu Land Use and Land Cover (LULC)?

Dalam penginderaan jauh (_remote sensing_) dan tata ruang wilayah, istilah **Land Cover** dan **Land Use** memiliki perbedaan mendasar yang saling melengkapi:

- **Land Cover (Penutupan Lahan):**
  Mengacu pada kenampakan fisik biofisik permukaan bumi yang terekam langsung oleh sensor satelit tanpa mempertimbangkan fungsi ekonominya.
  _Contoh:_ Tutupan vegetasi (Hutan Mangrove, Hutan Non-Mangrove), badan air (laut, sungai, waduk), atau material fisik permukaan bumi (bangunan, beton, tanah terbuka).

- **Land Use (Penggunaan Lahan):**
  Mengacu pada pemanfaatan dan fungsi sosial-ekonomi permukaan bumi oleh aktivitas manusia untuk tujuan tertentu.
  _Contoh:_ Area pertanian/sawah, permukiman warga, kawasan industri, pertambangan, dan area rekreasi.

---

### 1.2 Tujuan Analisis

Proyek analisis ini dirancang untuk mencapai 4 tujuan utama:

1. **Identifikasi & Pemetaan Otomatis:**
   Memetakan dan mengklasifikasikan tutupan lahan di wilayah kajian (Provinsi Jawa Timur) secara akurat dan otomatis menggunakan citra satelit optik Sentinel-2 multiband.

2. **Diferensiasi Khusus Hutan Mangrove:**
   Membedakan secara spesifik vegetasi **Hutan Mangrove** dengan **Hutan Non-Mangrove** dan **Pertanian/Sawah** berdasarkan respon pantulan spektral khas pada gelombang Shortwave Infrared (**SWIR / B11**), Near Infrared (**NIR / B08**), dan Red (**B04**).

3. **Verifikasi Visual Overlay:**
   Menampilkan hasil klasifikasi lahan dalam bentuk _interactive layer overlay_ (transparan) di atas citra satelit dasar (_basemap_ Google Satellite / Terrain) untuk memverifikasi secara visual kebenaran label klasifikasi di lapangan.

4. **Deployment Aplikasi Interaktif:**
   Membangun aplikasi web interaktif berbasis **Streamlit** untuk menyajikan peta klasifikasi LULC Jawa Timur, statistik luas area, dan evaluasi performa model secara intuitif bagi pemangku kepentingan (_stakeholders_).

---

### 1.3 Ruang Lingkup & 5 Kelas Tutupan Lahan

Untuk memodelkan kondisi fisiografis Jawa Timur, proyek ini mengklasifikasikan 5 kelas tutupan lahan pada 276 sampel polygon valid di `data/geojson/jawa_timur.geojson`. Dua fitur dengan geometri degeneratif dihapus dan lima geometri self-intersect diperbaiki; semua geometri sumber disimpan di arsip. Distribusi sampel tidak seimbang: Sawah (52), Pemukiman (62), Hutan Non-Mangrove (64), Hutan Mangrove (42), dan Air (56). Label kanonis yang digunakan adalah `Sawah`, `Pemukiman`, `Hutan non mangrove`, `Hutan mangrove`, dan `Air`.

| No  | Kelas LULC               | Deskripsi Karakteristik Fisikal                          | Respon Spektral Kunci                                      |
| :-: | :----------------------- | :------------------------------------------------------- | :--------------------------------------------------------- |
|  1  | **Sawah / Pertanian**    | Area bercocok tanam musiman dan tanah olahan pertanian   | Fluktuasi kelembapan tanah & pantulan vegetasi sedang      |
|  2  | **Pemukiman / Built-Up** | Bangunan perumahan, gedung, jalan, dan material buatan   | Respon SWIR & NDBI tinggi, NIR relatif rendah              |
|  3  | **Hutan Non-Mangrove**   | Vegetasi pepohonan dataran tinggi dan daratan            | Reflektansi NIR (B08) sangat tinggi, SWIR rendah           |
|  4  | **Hutan Mangrove**       | Vegetasi pesisir dan muara yang terpengaruh pasang surut | NIR tinggi namun SWIR/NDVI terpengaruh kelembapan air laut |
|  5  | **Air (Water Bodies)**   | Laut, sungai, waduk, dan tambak air                      | Penyerapan kuat pada NIR/SWIR, NDWI tinggi                 |

---

### 1.4 Pentingnya Diferensiasi Spektral Mangrove

Hutan Mangrove memiliki peran ekologis vital sebagai benteng alami dari abrasi pantai dan penyerap karbon (_carbon sink_). Namun secara spektral pada komposit warna visual (RGB), mangrove sering kali terlihat mirip dengan hutan daratan (Non-Mangrove) atau lahan pertanian subur.

Dengan memanfaatkan citra **Sentinel-2**, perbedaan ini dapat diselesaikan melalui kombinasi band spektral:

- **Band 11 (SWIR1):** Sensitif terhadap tingkat kebasahan (_moisture_) tanah di bawah kanopi mangrove.
- **Band 8 (NIR):** Membedakan kerapatan sel mesofil daun vegetasi sehat.
- **Indeks Spektral (NDVI, NDWI, NDBI):** Memperkuat marjin pemisah (_decision boundary_) antar kelas pada algoritma Machine Learning (seperti Random Forest).

---
