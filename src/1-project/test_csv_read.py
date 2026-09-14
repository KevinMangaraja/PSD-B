#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Setup
start_date = "2025-08-30"
end_date = "2026-08-30"

valid_pollutants = ['NO2', 'CO', 'SO2', 'CH4']
print("="*70)
print("MEMBACA DATA DARI CSV LOKAL")
print("="*70)

csv_file_path = Path('../../data/data_polutan_telukdalam_365_hari-Data Harian.csv')

if not csv_file_path.exists():
    print(f"Warning: File CSV tidak ditemukan di {csv_file_path}")
    csv_file_path = None
else:
    print(f"✓ File CSV ditemukan: {csv_file_path.resolve()}")

data_sets = {}
downloaded_files = {}

# Function definition
def process_pollutant(pollutant, csv_file_path):
    try:
        print(f"\nMEMPROSES: {pollutant}")
        df = pd.read_csv(csv_file_path)
        print(f"CSV berhasil dibaca ({len(df)} baris)")
        
        column_name = f"{pollutant} (Harian)"
        if column_name in df.columns:
            data_values = df[column_name].values
            print(f"Kolom '{column_name}' ditemukan")
        else:
            matching_cols = [col for col in df.columns if pollutant in col and "Harian" in col]
            if matching_cols:
                data_values = df[matching_cols[0]].values
                print(f"Menggunakan kolom: {matching_cols[0]}")
            else:
                raise ValueError(f"Kolom untuk {pollutant} tidak ditemukan di CSV")
        
        ds = xr.Dataset({pollutant: (('time',), data_values)})
        print(f"Dataset berhasil dibuat dengan {len(data_values)} data points")
        return ds
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Process data
if csv_file_path and csv_file_path.exists():
    print("\n" + "="*70)
    print("MEMBACA DATA DARI CSV")
    print("="*70)
    
    for pollutant in valid_pollutants:
        try:
            print(f"\n>>> Memproses: {pollutant}")
            dataset = process_pollutant(pollutant, str(csv_file_path))
            
            if dataset is not None:
                data_sets[pollutant] = dataset
                downloaded_files[pollutant] = pollutant
                print(f"✅ {pollutant}: Data CSV berhasil dibaca!")
                print(f"   Total {len(dataset['time'])} data points")
        except Exception as e:
            print(f"\n❌ ERROR untuk {pollutant}: {str(e)}")

print("\n" + "="*70)
print(f"✅ Total pollutant siap untuk analisis: {len(data_sets)}")
print(f"📊 Data dari CSV: {len(downloaded_files)}")
for pollutant in data_sets.keys():
    data_type = "CSV" if pollutant in downloaded_files else "SIMULASI"
    print(f"  - {pollutant} ({data_type}): {len(data_sets[pollutant]['time'])} data points")
print("="*70)

# Test: display first few values of each pollutant
print("\n" + "="*70)
print("SAMPLE DATA (first 5 values)")
print("="*70)
for pollutant, ds in data_sets.items():
    values = ds[pollutant].values[:5]
    print(f"{pollutant}: {values}")

# Check for missing values
print("\n" + "="*70)
print("CEK MISSING VALUES (DATA TETAP ORIGINAL)")
print("="*70)
for pollutant, ds in data_sets.items():
    values = ds[pollutant].values
    missing = np.isnan(values).sum()
    print(f"{pollutant}: Missing values = {missing} (Total = {len(values)})")
print("="*70)
