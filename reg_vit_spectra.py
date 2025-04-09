#!/usr/bin/env python

# -----------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Luis Felipe Strano Moraes
#
# This file is part of Spectromer
# For license terms, see the LICENSE file in the root of this repository.
# -----------------------------------------------------------------------------

import argparse
from datasets import load_dataset, DatasetDict
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTModel, TrainingArguments, Trainer
from torch import nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file as safetensors_load_file
from huggingface_hub import create_repo, HfApi
import os
import json
import random
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datasets import Dataset, DatasetDict, ClassLabel
import torchvision.transforms as T
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy import stats

# Define the mapping of integer values to string values
model_mapping = {
            0: "facebook/dino-vitb16",
            1: "facebook/dinov2-large", # requires image of 518*518
            2: "facebook/dino-vits16",
            3: "google/vit-large-patch32-384", # requires image of 384*384
            4: "google/vit-base-patch16-224",
}

os.environ["WANDB_PROJECT"] = "spectromer-regression"
os.environ["WANDB_LOG_MODEL"] = "false"

use_cpu_only = os.getenv("USE_CPU_ONLY", "0")  # Default to "0" (false)
device = torch.device("cpu" if use_cpu_only == "1" else ("cuda" if torch.cuda.is_available() else "cpu"))

class ViTRegressionModel(nn.Module):
    def __init__(self, pretrained_model='facebook/dino-vitb16', freeze_weights=False, num_layers=1, hidden_size=512):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model)

        if freeze_weights:
            for param in self.vit.parameters():
                param.requires_grad = False

        layers = []
        input_size = self.vit.config.hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, pixel_values, labels=None):
        pixel_values = pixel_values.to(device)
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :] # take the [CLS] token
        values = self.classifier(cls_output)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fct = nn.MSELoss()
            loss = loss_fct(values.view(-1), labels.view(-1))

        return (loss, values) if loss is not None else values

# Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.float32)}  # Return dict for Trainer

def train_model(opt):
    debug(opt, f"Starting regression finetuning on the {opt.label_field} field")
    imagedir = './data/' + opt.imagedir
    datasource = opt.dataset.split("-")[0]
    datasource_size = opt.dataset.split("-")[1]
    if (datasource == "lamost" or datasource == "slomost"):
        lamostdb = pd.read_csv(os.path.join("./data", 'meta-lamost-' + opt.dataset.split("-")[1] + '.csv'))
        debug(opt, f"Read lamost metadata file for {opt.dataset}")
        if (opt.label_field != 'z'):
            # For fields other than redshift, we want to consider only stars for now
            lamostdb = lamostdb[lamostdb['class'] == 'STAR']
        for column in lamostdb.columns:
            if pd.api.types.is_numeric_dtype(lamostdb[column]):
                lamostdb[column] = lamostdb[column].fillna(0)
            else:
                lamostdb[column] = lamostdb[column].fillna('null')
        debug(opt, f"Dataframe cleaned of missing values and complex characters")
    if (datasource == "sdss" or datasource == "slomost"):
        sdssdb = pd.read_csv(os.path.join("./data", 'meta-sdss-' + opt.dataset.split("-")[1] + '.csv'))
        debug(opt, f"Read sdss metadata file for {opt.dataset}")
        if (opt.label_field != 'z'):
            # For fields other than redshift, we want to consider only stars for now
            sdssdb = sdssdb[sdssdb['class'] == 'STAR']
        sdssdb.rename(columns={'elodieTEff': 'teff', 'elodieLogG': 'logg', 'elodieFeH': 'feh'}, inplace=True)
        for column in sdssdb.columns:
            if pd.api.types.is_numeric_dtype(sdssdb[column]):
                sdssdb[column] = sdssdb[column].fillna(0)
            else:
                sdssdb[column] = sdssdb[column].fillna('null')
        debug(opt, f"Dataframe cleaned of missing values and complex characters")
    if (datasource == "desi"):
        desidb = pd.read_csv(os.path.join("./data", "meta-desi-" + opt.dataset.split("-")[1] + '.csv'))
        debug(opt, f"Read desi metadata file for {opt.dataset}")
        for column in desidb.columns:
            if pd.api.types.is_numeric_dtype(desidb[column]):
                desidb[column] = desidb[column].fillna(0)
            else:
                desidb[column] = desidb[column].fillna('null')
        debug(opt, f"Dataframe cleaned of missing values and complex characters")
 
    # Update the DataFrame with filenames and gather available image files, then cleanup the columns
    required_columns = [ "filename", opt.label_field]
    if (datasource == "lamost" or datasource == "slomost"):
        lamostdb['filename'] = lamostdb.apply(
                lambda row: f'{row["obsdate"].replace("-","")}/{row["planid"]}/spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.png', axis=1
                )
        lamostdb = lamostdb[required_columns]
        if opt.mega:
            lamostdb_overlapsplit = lamostdb.copy()
            lamostdb_overlapsplit['filename'] = lamostdb_overlapsplit['filename'].str.replace('.png', '-overlapsplit.png', regex=False)

            lamostdb_map2d = lamostdb.copy()
            lamostdb_map2d['filename'] = lamostdb_map2d['filename'].str.replace('.png', '-map2d.png', regex=False)

            lamostdb = pd.concat([lamostdb, lamostdb_overlapsplit, lamostdb_map2d], ignore_index=True)

    if (datasource == "sdss" or datasource == "slomost"):
        sdssdb['filename'] = sdssdb.apply(
                lambda row: f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.png', axis=1
                )
        sdssdb = sdssdb[required_columns]
        if opt.mega:
            sdssdb_overlapsplit = sdssdb.copy()
            sdssdb_overlapsplit['filename'] = sdssdb_overlapsplit['filename'].str.replace('.png', '-overlapsplit.png', regex=False)

            sdssdb_map2d = sdssdb.copy()
            sdssdb_map2d['filename'] = sdssdb_map2d['filename'].str.replace('.png', '-map2d.png', regex=False)

            sdssdb = pd.concat([sdssdb, sdssdb_overlapsplit, sdssdb_map2d], ignore_index=True)

    if (datasource == "desi"):
        desidb['targetid'] = desidb['targetid'].astype(str)
        desidb['filename'] = desidb.apply(
                lambda row: f'spec-desi-GALAXY-UNKNOWN-{row["targetid"]}.png', axis=1
                )
        desidb = desidb[required_columns]
        if opt.mega:
            desidb_overlapsplit = desidb.copy()
            desidb_overlapsplit['filename'] = desidb_overlapsplit['filename'].str.replace('.png', '-overlapsplit.png', regex=False)

            desidb_map2d = desidb.copy()
            desidb_map2d['filename'] = desidb_map2d['filename'].str.replace('.png', '-map2d.png', regex=False)

            desidb = pd.concat([desidb, desidb_overlapsplit, desidb_map2d], ignore_index=True)

    if (datasource == "lamost"):
        spectradb = lamostdb
    elif (datasource == "sdss"):
        spectradb = sdssdb
    elif (datasource == "slomost"):
        spectradb = pd.concat([lamostdb, sdssdb], ignore_index=True)
    else: # datasource == "desi"
        spectradb = desidb
    debug(opt, f"Image filenames {len(spectradb)} added to the dataframe")
    # Check if files exist using vectorized operation and filter the DataFrame
    spectradb['file_exists'] = spectradb['filename'].apply(lambda f: os.path.isfile(os.path.join(imagedir, f)))
    # Keep only rows where the file exists
    spectradb = spectradb[spectradb['file_exists']].drop(columns='file_exists')
    debug(opt, f"Missing filenames dropped")
    image_files = spectradb['filename'].tolist()
    # Get the label values for regression
    reg_values = spectradb[opt.label_field].tolist()
    debug(opt, f"Found {len(image_files)} image files.")

    # Prepare the dataset as a list of dictionaries
    data = [{"image_path": os.path.join(imagedir, img), "labels": label} for img, label in zip(image_files, reg_values)]

    # Split the Dataset
#    if (datasource_size == "big"):
#        train_val_data, test_data = train_test_split(data, test_size=0.05, random_state=opt.random_seed)
#        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=opt.random_seed)
#    else:
#        train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=opt.random_seed)
#        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=opt.random_seed)

    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=opt.random_seed)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=opt.random_seed)
    # Get max value
    train_values = [item['labels'] for item in train_data]
    val_values = [item['labels'] for item in val_data]
    test_values = [item['labels'] for item in test_data]
    max_value = max(train_values + val_values + test_values)
    print('Max Value:', max_value)
    for item in train_data:
        item['labels'] = item['labels'] / max_value
    for item in val_data:
        item['labels'] = item['labels'] / max_value
    for item in test_data:
        item['labels'] = item['labels'] / max_value

    # Define a transform pipeline for the images
    transform_pipeline = T.Compose([
        T.Resize((224, 224)),  # Resize to the size expected by ViT
        T.ToTensor(),          # Convert PIL image to tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    train_values = [item['labels'] for item in train_data]
    train_images = [item['image_path'] for item in train_data]
    val_values = [item['labels'] for item in val_data]
    val_images = [item['image_path'] for item in val_data]
    test_values = [item['labels'] for item in test_data]
    test_images = [item['image_path'] for item in test_data]
 
    # Convert to Hugging Face Dataset
    debug(opt, f"Converting to Hugging Face Dataset")
    train_dataset = CustomDataset(train_images, train_values, transform=transform_pipeline)
    val_dataset = CustomDataset(val_images, val_values, transform=transform_pipeline)
    test_dataset = CustomDataset(test_images, test_values, transform=transform_pipeline)

    if opt.pretrained_model_dir:
        pretrained_model_path = opt.pretrained_model_dir
        print(f"Loading pre-trained model from {pretrained_model_path}")
    else:
        pretrained_model_path = model_mapping[opt.model]
        print(f"Using default pre-trained model {pretrained_model_path}")

    debug(opt, f"Moving model to device")
    model = ViTRegressionModel(pretrained_model=pretrained_model_path, freeze_weights=opt.freeze_weights, num_layers=opt.num_layers, hidden_size=opt.hidden_size).to(device)
    print(model)

    use_fp16=False
    if (opt.fp16):
        use_fp16=True
    training_args = TrainingArguments(
        output_dir='output-reg-' + opt.experiment,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=opt.batch_size,
        num_train_epochs=opt.num_train_epochs,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        save_steps=10,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
        resume_from_checkpoint=True,
        report_to="wandb",
        run_name=opt.experiment,
        seed=opt.random_seed,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model='r2',
        greater_is_better=True,
        eval_accumulation_steps=32,
    )

    # Add logging to inspect the model outputs and labels
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        print(f"Shape of preds: {preds.shape}, dtype: {preds.dtype}")
        print(f"Shape of labels: {labels.shape}, dtype: {labels.dtype}")
        random_indices = np.random.choice(len(preds), size=5, replace=False)
        random_preds = preds[random_indices]
        random_labels = labels[random_indices]
        with np.printoptions(suppress=True, precision=6):
            debug(opt, f"Random Predictions: {random_preds}")
            debug(opt, f"Random Labels: {random_labels}")
        preds = preds.squeeze()
        labels = labels.flatten()
        print(f"Reshaped of preds: {preds.shape}, dtype: {preds.dtype}")
        print(f"Reshaped of labels: {labels.shape}, dtype: {labels.dtype}")
        assert preds.shape == labels.shape, f"Shape mismatch: preds {preds.shape}, labels {labels.shape}"
        mse = ((preds - labels) ** 2).mean().item()
        r2 = r2_score(labels, preds)
        return {"mse": mse, "r2": r2}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the model and config
    if opt.save_model:
        model_save_dir = os.path.join(training_args.output_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_save_path = os.path.join(model_save_dir, 'pytorch_model.bin')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        # Save the config (max_value)
        config = {'max_value': max_value}
        config_path = os.path.join(model_save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        print(f"Config saved to {config_path}")

    if (opt.deterministic):
        model.eval() # disables dropout and batch normalization
    eval_results = trainer.evaluate(test_dataset)
    print(f"Evaluation results: {eval_results}")

    # Extract the predictions and labels from the test set
    predictions, labels, _ = trainer.predict(test_dataset)
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Rescale predictions and labels back to original scale
    predictions = predictions * max_value
    labels = labels * max_value

    # Compute residuals
    residuals = predictions - labels
    perc_residuals = ((predictions - labels) / labels) * 100

    lower_percentile = np.percentile(residuals, 5)  # 5th percentile
    upper_percentile = np.percentile(residuals, 95)  # 95th percentile

    mask = (residuals >= lower_percentile) & (residuals <= upper_percentile)
    residuals_filtered = residuals[mask]
    perc_residuals_filtered = perc_residuals[mask]
    perc_mask = (perc_residuals_filtered > 200) | (perc_residuals_filtered < -200)
    labels_filtered = labels[mask]

    r2_test = r2_score(labels, predictions)
    print(f"R2 Score for the test dataset: {r2_test}")

    # Optionally, save predictions and labels to a CSV
    if opt.output_predictions:
        results = pd.DataFrame({'image_path': test_images, 'prediction': predictions, 'label': labels})
        results.to_csv(opt.output_predictions, index=False)
        print(f"Predictions saved to {opt.output_predictions}")

    # Plot the residuals
    plt.figure()
    plt.scatter(labels, residuals, linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Residuals")
    plt.title("Residual Plot - " + opt.experiment)
    plt.legend()
    wandb.log({"train_residuals-values": wandb.Image(plt)})
    plt.close()

    plt.figure()
    plt.scatter(labels_filtered, residuals_filtered, linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Residuals Filtered (5% - 95%)")
    plt.title("Residual Filtered Plot - " + opt.experiment)
    plt.legend()
    wandb.log({"train_residuals-valuesfiltered": wandb.Image(plt)})
    plt.close()

    # Plot the residuals percentages
    plt.figure()
    plt.scatter(labels, perc_residuals, linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Percentage Residuals (%)")
    plt.title("Percentage Residual Plot - " + opt.experiment)
    plt.legend()
    wandb.log({"train_residuals-percentage": wandb.Image(plt)})
    plt.close()

    plt.figure()
    plt.scatter(labels_filtered[~perc_mask], perc_residuals_filtered[~perc_mask], linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Percentage Residuals Filtered (%)")
    plt.ylim(-200, 200)
    plt.title("Percentage Residual Filtered Plot - " + opt.experiment)
    plt.legend()
    wandb.log({"train_residuals-percentagefiltered": wandb.Image(plt)})
    plt.close()

def evaluate_model(opt):
    # Load the model
    model_dir = opt.load_model_dir
    model_path = os.path.join(model_dir, 'pytorch_model.bin')
    if not os.path.isfile(model_path):
        print(f"Model file not found at {model_path}")
        return
    model = ViTRegressionModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load the config
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(config_path):
        print(f"Config file not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    max_value = config['max_value']
    print(f"Config loaded from {config_path}")

    # Prepare the dataset (same as in training)
    imagedir = './data/' + opt.imagedir
    datasource = opt.dataset.split("-")[0]
    spectradb = pd.read_csv(os.path.join("./data", 'meta-' + opt.dataset + '.csv'))
    for column in spectradb.columns:
        if pd.api.types.is_numeric_dtype(spectradb[column]):
            spectradb[column] = spectradb[column].fillna(0)
        else:
            spectradb[column] = spectradb[column].fillna('null')
    debug(opt, f"Dataframe cleaned of missing values and complex characters")
    # Update the DataFrame with filenames and gather available image files
    if (datasource == "lamost"):
        spectradb['filename'] = spectradb.apply(
                lambda row: f'{row["obsdate"].replace("-","")}/{row["planid"]}/spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.png', axis=1
                )
    elif (datasource == "sdss"):
        spectradb['filename'] = spectradb.apply(
                lambda row: f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.png', axis=1
                )
    else:
        spectradb['targetid'] = spectradb['targetid'].astype(str)
        spectradb['filename'] = spectradb.apply(
                lambda row: f'spec-desi-GALAXY-UNKNOWN-{row["targetid"]}.png', axis=1
                )
    debug(opt, f"Image filenames added to the dataframe")
    # Check if files exist using vectorized operation and filter the DataFrame
    spectradb['file_exists'] = spectradb['filename'].apply(lambda f: os.path.isfile(os.path.join(imagedir, f)))
    # Keep only rows where the file exists
    spectradb = spectradb[spectradb['file_exists']].drop(columns='file_exists')
    debug(opt, f"Missing filenames dropped")
    image_files = spectradb['filename'].tolist()
    # Get the redshift values for regression
    reg_values = spectradb[opt.label_field].tolist()
    debug(opt, f"Found {len(image_files)} image files.")

    # Prepare the dataset as a list of dictionaries
    data = [{"image_path": os.path.join(imagedir, img), "labels": label} for img, label in zip(image_files, reg_values)]

    # Normalize labels
    for item in data:
        item['labels'] = item['labels'] / max_value

    # Define the same transform pipeline as in training
    transform_pipeline = T.Compose([
        T.Resize((224, 224)),  # Resize to the size expected by ViT
        T.ToTensor(),          # Convert PIL image to tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    values = [item['labels'] for item in data]
    images = [item['image_path'] for item in data]

    # Prepare the dataset
    eval_dataset = CustomDataset(images, values, transform=transform_pipeline)

    # Create DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False)

    # Run inference
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(pixel_values)
            values = outputs.detach().cpu().numpy().flatten()
            predictions.extend(values)
            labels.extend(batch['labels'].numpy())

    # Rescale predictions and labels back to original scale
    predictions = np.array(predictions) * max_value
    labels = np.array(labels) * max_value

    # Compute metrics
    mse = ((predictions - labels) ** 2).mean().item()
    r2 = r2_score(labels, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Compute residuals
    residuals = predictions - labels
    perc_residuals = ((predictions - labels) / labels) * 100

    lower_percentile = np.percentile(residuals, 5)  # 5th percentile
    upper_percentile = np.percentile(residuals, 95)  # 95th percentile

    mask = (residuals >= lower_percentile) & (residuals <= upper_percentile)
    residuals_filtered = residuals[mask]
    perc_residuals_filtered = perc_residuals[mask]
    perc_mask = (perc_residuals_filtered > 200) | (perc_residuals_filtered < -200)
    labels_filtered = labels[mask]

    # Plot the residuals
    plt.figure()
    plt.scatter(labels, residuals, linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Residuals")
    plt.title("Residual Plot - Evaluation")
    plt.legend()
    plt.savefig('eval_residuals-values-' + opt.experiment + '.png')
    plt.close()

    plt.figure()
    plt.scatter(labels_filtered, residuals_filtered, linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Residuals Filtered (5% - 95%)")
    plt.title("Residual Filtered Plot - Evaluation")
    plt.legend()
    plt.savefig('eval_residuals-valuesfiltered-' + opt.experiment + '.png')
    plt.close()

    # Plot the residuals percentages
    plt.figure()
    plt.scatter(labels, perc_residuals, linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Percentage Residuals (%)")
    plt.title("Percentage Residual Plot - Evaluation")
    plt.legend()
    plt.savefig('eval_residuals-percentage-' + opt.experiment + '.png')
    plt.close()

    plt.figure()
    plt.scatter(labels_filtered[~perc_mask], perc_residuals_filtered[~perc_mask], linewidth=0, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', label="Zero residuals")
    plt.xlabel("True Values (Labels)")
    plt.ylabel("Percentage Residuals Filtered (%)")
    plt.ylim(-200, 200)
    plt.title("Percentage Residual Filtered Plot - Evaluation")
    plt.legend()
    plt.savefig('eval_residuals-percentagefiltered-' + opt.experiment + '.png')
    plt.close()

    # Optionally, save predictions and labels to a CSV
    if opt.output_predictions:
        results = pd.DataFrame({'image_path': images, 'prediction': predictions, 'label': labels})
        results.to_csv(opt.output_predictions, index=False)
        print(f"Predictions saved to {opt.output_predictions}")

def debug(opt, message):
    if opt.debug:
        print(message)

def run(opt):
    debug(opt, f"Starting with device {device}")
    if opt.do_eval:
        if not opt.load_model_dir:
            print("Please specify --load_model_dir for evaluation.")
        else:
            evaluate_model(opt)
    elif opt.do_train:
        train_model(opt)
    else:
        print("Please specify --do_train or --do_eval")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sdss-small', type=str,
                    help='Metadata of dataset to use, will look for corresponding ./data/meta-<EXPERIMENT>.csv file')
    parser.add_argument('--imagedir', default='png-sdss-small', type=str,
                    help='Image dataset to use, will look for corresponding ./data/<IMAGES> directory with PNG files')
    parser.add_argument('--experiment', default=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), type=str,
                    help='Name to use for this experiment')
    parser.add_argument('--num_train_epochs', default=30, type=int, help='Number of epochs to use')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--debug', action='store_true', help='A debugging flag to be used when testing.')
    parser.add_argument('--deterministic', action='store_true', help='Attempt to do a deterministic run')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed to use')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size to be used for training')
    parser.add_argument('--model', default=0, type=int, choices=model_mapping.keys(), help='Choice between pretrained models to base on')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model after training')
    parser.add_argument('--pretrained_model_dir', type=str, default=None, help='Path to pretrained model if desired')

    # Evaluation arguments
    parser.add_argument('--do_train', action='store_true', help='Run training')
    parser.add_argument('--do_eval', action='store_true', help='Run evaluation')
    parser.add_argument('--load_model_dir', type=str, default=None, help='Directory containing the saved model to load for evaluation')
    parser.add_argument('--output_predictions', type=str, default=None, help='Path to save predictions CSV file')

    parser.add_argument('--label_field', choices=['z', 'teff', 'logg', 'feh'], default='z', help='Which metadata field to use as label: z (default), teff, logg or feh')

    parser.add_argument('--mega', action='store_true', help='Look for images with different plot types')

    parser.add_argument('--freeze_weights', action='store_true', help='Freeze pretrained model weights')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the final added head')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of added layers')

    opt = parser.parse_args()
    run(opt)

