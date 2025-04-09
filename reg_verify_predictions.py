#!/usr/bin/env python

# -----------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Luis Felipe Strano Moraes
#
# This file is part of Spectromer
# For license terms, see the LICENSE file in the root of this repository.
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import re
from scipy.stats import norm
from properscoring import crps_gaussian
from sklearn.metrics import mean_squared_error

# metadata directory
METADATA_DIR = "./data"

if len(sys.argv) != 2:
    print("Usage: python script_name.py predictions.csv")
    sys.exit(1)

input_csv = sys.argv[1]

try:
    df = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"Error: File '{input_csv}' not found.")
    sys.exit(1)

# Check and merge 'snMedian' if not present
if 'snMedian' not in df.columns:
    print("snMedian column not found. Merging from metadata files.")

    def extract_info(image_path):
        filename = os.path.basename(image_path)
        parts = filename.replace('.png', '').split('-')
        info = {'class': 'Unknown', 'datasource': 'Unknown', 'unique_id': None, 'size': 'big'}
        if len(parts) >= 5:
            info['datasource'] = parts[1]
            info['class'] = parts[2]
            match = re.search(r'_(small|medium|big)', input_csv)
            info['size'] = match.group(1) if match else 'big'
            info['unique_id'] = parts[4]
        return pd.Series(info)

    df = df.join(df['image_path'].apply(extract_info))

    metadata_files = {}
    datasource_list = df['datasource'].unique()
    size = df['size'].iloc[0]

    for datasource in datasource_list:
        metadata_filename = f'meta-{datasource}-{size}.csv'
        metadata_filepath = os.path.join(METADATA_DIR, metadata_filename)
        try:
            metadata_df = pd.read_csv(metadata_filepath).fillna(0)
            metadata_files[datasource] = metadata_df
        except FileNotFoundError:
            print(f"Error: Metadata file '{metadata_filepath}' not found.")
            sys.exit(1)

    def get_sn_median(row):
        datasource = row['datasource']
        unique_id = row['unique_id']
        metadata_df = metadata_files.get(datasource)
        if metadata_df is not None and unique_id is not None:
            try:
                unique_id_int = int(unique_id)
                match = metadata_df[metadata_df['obsid' if datasource == 'lamost' else 'specObjID'] == unique_id_int]
            except ValueError:
                match = metadata_df[metadata_df['obsid' if datasource == 'lamost' else 'specObjID'] == unique_id]
            return match.iloc[0]['snMedian'] if not match.empty else np.nan

    df['snMedian'] = df.apply(get_sn_median, axis=1)
    df.to_csv(input_csv, index=False)
    print(f"Updated CSV file saved with 'snMedian' column added.")

# Drop rows where 'label' is zero to avoid MAPE issues
df = df[df['label'] != 0]

# Calculate Errors
df['difference'] = df['prediction'] - df['label']
df['abs_percentage_error'] = np.abs(df['difference'] / df['label']) * 100

# Handle infinite values in MAPE (replace with NaN)
df['abs_percentage_error'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Store MAPE before capping
mape_uncapped = np.nanmean(df['abs_percentage_error'])  # Uncapped mean

# Cap MAPE at 500% to avoid extreme outliers
cap_threshold = 500
df['capped_abs_percentage_error'] = df['abs_percentage_error'].clip(upper=cap_threshold)

# Compute Mean and Median Absolute Percentage Error
mape_capped = np.mean(df['capped_abs_percentage_error'])  # Capped mean
mdape = np.median(df['abs_percentage_error'])  # Median for stability

# CRPS Calculation
sigma = np.std(df['difference'])
df['CRPS'] = crps_gaussian(df['label'], df['prediction'], sigma)
mean_crps = np.mean(df['CRPS'])

# RMSE Calculation
rmse = np.sqrt(mean_squared_error(df['label'], df['prediction']))

# Metrics by Class
crps_by_class = df.groupby('class')['CRPS'].mean()
rmse_by_class = df.groupby('class').apply(lambda x: np.sqrt(mean_squared_error(x['label'], x['prediction'])))
mape_by_class = df.groupby('class')['capped_abs_percentage_error'].mean()
mape_uncapped_by_class = df.groupby('class')['abs_percentage_error'].mean()
mdape_by_class = df.groupby('class')['abs_percentage_error'].median()

# Print results
print("\n=== Model Evaluation Metrics ===")
print(f"Mean CRPS Score: {mean_crps:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE - Uncapped): {mape_uncapped:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE - Capped at 500%): {mape_capped:.2f}%")
print(f"Median Absolute Percentage Error (MdAPE): {mdape:.2f}%\n")

print("=== Per-Class Metrics ===")
print("CRPS by Class:")
print(crps_by_class)
print("\nRMSE by Class:")
print(rmse_by_class)
print("\nMAPE by Class (Uncapped):")
print(mape_uncapped_by_class)
print("\nMAPE by Class (Capped at 500%):")
print(mape_by_class)
print("\nMdAPE by Class:")
print(mdape_by_class)

# Save Metrics to CSV
output_metrics_csv = os.path.splitext(input_csv)[0] + '_metrics.csv'
metrics_df = pd.DataFrame({
    'Class': crps_by_class.index,
    'CRPS': crps_by_class.values,
    'RMSE': rmse_by_class.values,
    'MAPE (Uncapped)': mape_uncapped_by_class.values,
    'MAPE (Capped)': mape_by_class.values,
    'MdAPE': mdape_by_class.values
    })

metrics_df.to_csv(output_metrics_csv, index=False)
print(f"\nMetrics saved to {output_metrics_csv}")

# Define SNR bins
sn_bins = [0, 1, 2, 5, 7, 10, 20, 30, 40, 50, np.inf]
sn_labels = ['0-1', '1-2', '2-5', '5-7', '7-10', '10-20', '20-30', '30-40', '40-50', '50+']
df['sn_bin'] = pd.cut(df['snMedian'], bins=sn_bins, labels=sn_labels, right=False)

# Success Criteria Analysis
sn_results_dict = {}
results_dict = {}

# comes from Ross 2020
criteria_galaxy = 0.00334
criteria_star = 0.00334
criteria_qso = 0.01001

df.loc[df['class'] == 'GALAXY', f'meets_{criteria_galaxy}'] = (
            df['difference'].abs() < criteria_galaxy * (1 + df['label'])
            )
df.loc[df['class'] == 'STAR', f'meets_{criteria_star}'] = (
            df['difference'].abs() < criteria_star * (1 + df['label'])
            )
df.loc[df['class'] == 'QSO', f'meets_{criteria_qso}'] = (
        df['difference'].abs() < criteria_qso * (1 + df['label'])
        )

grouped_sn_galaxy = df[df['class'] == 'GALAXY'].groupby('sn_bin')[f'meets_{criteria_galaxy}'].mean() * 100
grouped_sn_star = df[df['class'] == 'STAR'].groupby('sn_bin')[f'meets_{criteria_star}'].mean() * 100
grouped_sn_qso = df[df['class'] == 'QSO'].groupby('sn_bin')[f'meets_{criteria_qso}'].mean() * 100

overall_success_galaxy = df[df['class'] == 'GALAXY'][f'meets_{criteria_galaxy}'].mean() * 100
overall_success_star = df[df['class'] == 'STAR'][f'meets_{criteria_star}'].mean() * 100
overall_success_qso = df[f'meets_{criteria_qso}'].mean() * 100

# Compute combined success rate without STAR
grouped_sn_galaxy_qso = df[df['class'].isin(['GALAXY', 'QSO'])].groupby('sn_bin')[['meets_0.00334', 'meets_0.01001']].mean().mean(axis=1) * 100
overall_success_all = df[['meets_0.00334', 'meets_0.01001']].mean().mean() * 100
overall_success_all_minus_star = df[df['class'] != 'STAR'][['meets_0.00334', 'meets_0.01001']].mean().mean() * 100

# Combine results
sn_results_df = pd.DataFrame({
    'sn_bin': sn_labels,
    'success GALAXIES': grouped_sn_galaxy,
    'success QSO': grouped_sn_qso,
    'success STAR': grouped_sn_star,
    'success GALAXY+QSO': grouped_sn_galaxy_qso,
    'success combined': df.groupby('sn_bin')[['meets_0.00334', 'meets_0.01001']].mean().mean(axis=1) * 100
    }).reset_index(drop=True)

print("\n# Overall Success Criteria Analysis")
print({
    'Criteria 0.3% GALAXY': overall_success_galaxy,
    'Criteria 0.3% STAR': overall_success_star,
    'Criteria 1.0% QSO': overall_success_qso,
    'Criteria ALL': overall_success_all,
    'Criteria ALL minus STARS': overall_success_all_minus_star
    })

output_snr_csv = os.path.splitext(input_csv)[0] + '_snr_analysis.csv'
sn_results_df.to_csv(output_snr_csv, index=False)

print(f"\nSNR Analysis saved to {output_snr_csv}")

# Print LaTeX-compatible table format
print("\nLaTeX Table Format:")
for _, row in sn_results_df.iterrows():
#        print(f"{row['sn_bin']} & {row['success GALAXIES']:.2f} & {row['success QSO']:.2f} & {row['success STAR']:.2f} & {row['success combined']:.2f} \\")
        print(f"{row['sn_bin']} & {row['success combined']:.2f} \\\\ \\hline")

