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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score
import seaborn as sns
import re

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

# Check if 'snMedian' is already in the DataFrame
if 'snMedian' in df.columns:
    print("snMedian column found in the CSV file. Reusing existing values.")
else:
    print("snMedian column not found. Merging snMedian from metadata files.")
    # Extract 'class', 'datasource', 'unique_id', and 'size' from the 'image_path'
    def extract_info(image_path):
        # Assuming the filename formats are:
        # SDSS: 'spec-sdss-class-subClass-specObjID.png'
        # LAMOST: 'spec-lamost-class-subClass-obsid.png'
        filename = os.path.basename(image_path)
        parts = filename.replace('.png', '').split('-')
        info = {'class_extracted': 'Unknown', 'datasource': 'Unknown', 'unique_id': None, 'size': 'big'}
        if len(parts) >= 5:
            info['datasource'] = parts[1]
            info['class_extracted'] = parts[2]
            # Extract 'size' from the experiment name if available
            match = re.search(r'_(small|medium|big)', input_csv)
            if match:
                info['size'] = match.group(1)
            else:
                info['size'] = 'big'  # Default to 'big' if not found
            if info['datasource'] == 'sdss':
                info['unique_id'] = parts[4]  # specObjID for SDSS
            elif info['datasource'] == 'lamost':
                info['unique_id'] = parts[4]  # obsid for LAMOST
        return pd.Series(info)

    # Apply the extraction function to the dataframe
    df = df.join(df['image_path'].apply(extract_info))

    # Load metadata files
    metadata_files = {}
    datasource_list = df['datasource'].unique()
    size = df['size'].iloc[0]  # Assuming size is consistent across the dataset
    for datasource in datasource_list:
        metadata_filename = f'meta-{datasource}-{size}.csv'  # Including size in filename
        metadata_filepath = os.path.join(METADATA_DIR, metadata_filename)
        try:
            metadata_df = pd.read_csv(metadata_filepath)
            # Clean missing values
            for column in metadata_df.columns:
                if pd.api.types.is_numeric_dtype(metadata_df[column]):
                    metadata_df[column] = metadata_df[column].fillna(0)
                else:
                    metadata_df[column] = metadata_df[column].fillna('null')
            metadata_files[datasource] = metadata_df
        except FileNotFoundError:
            print(f"Error: Metadata file '{metadata_filepath}' not found.")
            sys.exit(1)

    # Merge 'snMedian' from metadata into the main dataframe
    def get_sn_median(row):
        datasource = row['datasource']
        unique_id = row['unique_id']
        metadata_df = metadata_files.get(datasource)
        if metadata_df is not None and unique_id is not None:
            if datasource == 'sdss':
                # Use 'specObjID' as the key for SDSS
                try:
                    unique_id_int = int(unique_id)
                    match = metadata_df[metadata_df['specObjID'] == unique_id_int]
                except ValueError:
                    match = metadata_df[metadata_df['specObjID'] == unique_id]
                if not match.empty:
                    return match.iloc[0]['snMedian']
            elif datasource == 'lamost':
                # Use 'obsid' as the key for LAMOST
                try:
                    unique_id_int = int(unique_id)
                    match = metadata_df[metadata_df['obsid'] == unique_id_int]
                except ValueError:
                    match = metadata_df[metadata_df['obsid'] == unique_id]
                if not match.empty:
                    return match.iloc[0]['snMedian']
        return np.nan

    df['snMedian'] = df.apply(get_sn_median, axis=1)

    # write the updated DataFrame back to the CSV file
    df.to_csv(input_csv, index=False)
    print(f"Updated CSV file saved with 'snMedian' column added.")

# drop rows where 'snMedian' could not be found
df = df.dropna(subset=['snMedian'])

# extract true labels and predicted labels
true_labels = df['label']
pred_labels = df['prediction']

# get the list of unique labels
labels = sorted(list(set(true_labels) | set(pred_labels)))

# compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# output the confusion matrix in text format
print("Confusion Matrix:")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# output classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, labels=labels))

# Compute overall accuracy
overall_accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Compute per-class recall
per_class_recall = recall_score(true_labels, pred_labels, labels=labels, average=None)
print("\nPer-Class Recall:")
for label, recall in zip(labels, per_class_recall):
    print(f"{label}: {recall:.4f}")

# Compute macro-averaged F1 score
macro_f1 = f1_score(true_labels, pred_labels, labels=labels, average='macro')
print(f"\nMacro-Averaged F1 Score: {macro_f1:.4f}")

# generate output filename by replacing .csv with .png
output_filename = os.path.splitext(input_csv)[0] + '.png'

# plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(output_filename)
print(f"Confusion matrix plot saved as {output_filename}")

# Additional SNR Analysis

# Define the new SNR bins without duplicate values
sn_bins = [0, 1, 2, 5, 7] + list(range(10, 100, 10)) + [np.inf]
sn_labels = ['0-1', '1-2', '2-5', '5-7'] + [f'{i}-{i+10}' for i in range(7, 90, 10)] + ['90+']

df['sn_bin'] = pd.cut(df['snMedian'], bins=sn_bins, labels=sn_labels, right=False)

# list to store results
sn_results = []

for sn_range in sn_labels:
    bin_df = df[df['sn_bin'] == sn_range]
    if not bin_df.empty:
        true_bin_labels = bin_df['label']
        pred_bin_labels = bin_df['prediction']
        accuracy = accuracy_score(true_bin_labels, pred_bin_labels)
        sn_results.append({'SNR Range': sn_range, 'Accuracy': accuracy * 100, 'Count': len(bin_df)})
    else:
        sn_results.append({'SNR Range': sn_range, 'Accuracy': np.nan, 'Count': 0})

# convert results to DataFrame
sn_results_df = pd.DataFrame(sn_results)

# output the results in a table format
print("\nAccuracy by SNR Range:")
print(sn_results_df.to_string(index=False))

# plot the accuracy by SNR range
plt.figure(figsize=(12, 8))
plt.bar(sn_results_df['SNR Range'], sn_results_df['Accuracy'])
plt.xlabel('SNR Range')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy by SNR Range')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.tight_layout()

sn_output_filename = os.path.splitext(input_csv)[0] + '_sn_accuracy.png'
plt.savefig(sn_output_filename)
print(f"SNR range accuracy plot saved as {sn_output_filename}")

