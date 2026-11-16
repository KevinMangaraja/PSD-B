# Klasifikasi Kerapian Alas Kaki

Pipeline menggunakan citra pada `proyek_sain_data/data/RAPI` dan
`proyek_sain_data/data/BERANTAKAN`.

Jalankan dari folder `10-project_citra`:

```powershell
python run_pipeline.py
```

Pipeline mengekstrak fitur visual RGB/HSV, grayscale, tepi, kuadran, dan
histogram gradien lokal HOG; membuat split berdasarkan kelompok pengambilan;
memilih model berdasarkan macro-F1 group CV; melatih ulang pada semua data
pengembangan; lalu menguji pada kelompok yang tidak dipakai selama pelatihan.

Split yang ditetapkan untuk dataset saat ini:

- Pengujian: `Kelompok 2` (15 citra per kelas)
- Pengembangan: seluruh kelompok lainnya

Model dipilih dengan 5-fold `StratifiedGroupKFold` pada data pengembangan.
Scaler dipelajari ulang pada setiap fold; kemudian model terpilih dan scaler
dilatih pada seluruh data pengembangan. Tidak ada kelompok yang dibagi lintas
split. Kandidat label yang nama filenya bertentangan dengan folder dicatat
untuk ditinjau, tanpa mengubah label otomatis. Artefak fitur dan scaler disimpan di
`3-data_preparation/data/processed`, model di `4-modeling/models`, salinan
deployment di `6-deployment`, dan metrik di `5-evaluation/results`.

Model yang dibandingkan adalah Extra Trees, Random Forest, Gradient Boosting,
SVC, dan Logistic Regression. Ini merupakan baseline machine learning klasik,
bukan CNN. Hasil uji perlu dibaca sebagai estimasi awal karena dataset hanya
memiliki tujuh kelompok pengambilan dan test akhir berasal dari satu kelompok.
