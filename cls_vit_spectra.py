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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict, ClassLabel
import torchvision.transforms as T
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy import stats
from rewrite_subclasses import rewrite_subclass_names

# Some parts used from https://github.com/TonyAssi/ImageRegression/blob/main/README.md

os.environ["WANDB_PROJECT"] = "spectromer-classifier"
os.environ["WANDB_LOG_MODEL"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTClassificationModel(nn.Module):
    def __init__(self, num_labels, pretrained_model='facebook/dino-vitb16'):
        super(ViTClassificationModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        pixel_values = pixel_values.to(device)
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits) if loss is not None else logits

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

        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.long)}  # Return dict for Trainer

def train_model(opt):
    imagedir = './data/' + opt.imagedir
    datasource = opt.dataset.split("-")[0]
    if (datasource == "lamost" or datasource == "slomost"):
        lamostdb = pd.read_csv(os.path.join("./data", 'meta-lamost-' + opt.dataset.split("-")[1] + '.csv'))
        debug(opt, f"Read metadata file for {opt.dataset}")
        for column in lamostdb.columns:
            if pd.api.types.is_numeric_dtype(lamostdb[column]):
                lamostdb[column] = lamostdb[column].fillna(0)
            else:
                lamostdb[column] = lamostdb[column].fillna('null')
        debug(opt, f"Dataframe cleaned of missing values and complex characters")
    if (datasource == "sdss" or datasource == "slomost"):
        sdssdb = pd.read_csv(os.path.join("./data", 'meta-sdss-' + opt.dataset.split("-")[1] + '.csv'))
        debug(opt, f"Read metadata file for {opt.dataset}")
        for column in sdssdb.columns:
            if pd.api.types.is_numeric_dtype(sdssdb[column]):
                sdssdb[column] = sdssdb[column].fillna(0)
            else:
                sdssdb[column] = sdssdb[column].fillna('null')
        debug(opt, f"Dataframe cleaned of missing values and complex characters")

    # Update the DataFrame with filenames and gather available image files, then cleanup the columns
    required_columns = [ "filename", "class", "subclass" ]
    if (datasource == "lamost" or datasource == "slomost"):
        lamostdb['filename'] = lamostdb.apply(
                lambda row: f'{row["obsdate"].replace("-","")}/{row["planid"]}/spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.png', axis=1
                )
        lamostdb = lamostdb[required_columns]
    if (datasource == "sdss" or datasource == "slomost"):
        sdssdb['filename'] = sdssdb.apply(
                lambda row: f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.png', axis=1
                )
        # Rename 'subClass' to 'subclass' for consistency
        sdssdb.rename(columns={'subClass': 'subclass'}, inplace=True)
        sdssdb = sdssdb[required_columns]
    debug(opt, f"Image filenames added to the dataframe")
    if (datasource == "lamost"):
        spectradb = rewrite_subclass_names(lamostdb)
    elif (datasource == "sdss"):
        spectradb = rewrite_subclass_names(sdssdb)
    elif (datasource == "slomost"):
        spectradb = rewrite_subclass_names(pd.concat([lamostdb, sdssdb], ignore_index=True))
    # Clean the labels
    spectradb[opt.label_field] = spectradb[opt.label_field].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    spectradb[opt.label_field] = spectradb[opt.label_field].fillna('Unknown')
    # Check if files exist using vectorized operation and filter the DataFrame
    spectradb['file_exists'] = spectradb['filename'].apply(lambda f: os.path.isfile(os.path.join(imagedir, f)))
    # Keep only rows where the file exists
    spectradb = spectradb[spectradb['file_exists']].drop(columns='file_exists')
    debug(opt, f"Missing filenames dropped")
    image_files = spectradb['filename'].tolist()
    # Get the labels for classification
    labels_list = spectradb[opt.label_field].tolist()
    debug(opt, f"Found {len(image_files)} image files.")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_list)
    num_labels = len(label_encoder.classes_)
    debug(opt, f"Number of classes: {num_labels}")
    # Save label mapping for future use
    label_mapping = dict(zip(range(num_labels), label_encoder.classes_))

    # Prepare the dataset as a list of dictionaries
    data = [{"image_path": os.path.join(imagedir, img), "labels": label} for img, label in zip(image_files, labels)]

    # Split the Dataset
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=opt.random_seed)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=opt.random_seed)

    # Define a transform pipeline for the images
    transform_pipeline = T.Compose([
        T.Resize((224, 224)),  # Resize to the size expected by ViT
        T.ToTensor(),          # Convert PIL image to tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    train_labels = [item['labels'] for item in train_data]
    train_images = [item['image_path'] for item in train_data]
    val_labels = [item['labels'] for item in val_data]
    val_images = [item['image_path'] for item in val_data]
    test_labels = [item['labels'] for item in test_data]
    test_images = [item['image_path'] for item in test_data]

    # Convert to Hugging Face Dataset
    debug(opt, f"Converting to Hugging Face Dataset")
    train_dataset = CustomDataset(train_images, train_labels, transform=transform_pipeline)
    val_dataset = CustomDataset(val_images, val_labels, transform=transform_pipeline)
    test_dataset = CustomDataset(test_images, test_labels, transform=transform_pipeline)

    if opt.pretrained_model_dir:
        pretrained_model_path = opt.pretrained_model_dir
        print(f"Loading pre-trained model from {pretrained_model_path}")
    else:
        pretrained_model_path = 'facebook/dino-vitb16'
        print(f"Using default pre-trained model {pretrained_model_path}")

    debug(opt, f"Moving model to device")
    model = ViTClassificationModel(num_labels=num_labels, pretrained_model=pretrained_model_path).to(device)

    use_fp16=False
    if (opt.fp16):
        use_fp16=True
    training_args = TrainingArguments(
        output_dir='output-cls-' + opt.experiment,
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
        metric_for_best_model='accuracy',
        greater_is_better=True,
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        debug(opt, f"Predictions: {preds[:5]}")
        debug(opt, f"Labels: {labels[:5]}")
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

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
        # Save the config (label mapping)
        config = {'label_mapping': label_encoder.classes_.tolist()}
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
    preds = np.argmax(predictions, axis=1)
    labels = labels.flatten()

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1 Score: {f1}")

    # Optionally, save predictions and labels to a CSV
    if opt.output_predictions:
        inverse_label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        preds_labels = [inverse_label_mapping[pred] for pred in preds]
        true_labels = [inverse_label_mapping[label] for label in labels]
        results = pd.DataFrame({'image_path': test_images, 'prediction': preds_labels, 'label': true_labels})
        results.to_csv(opt.output_predictions, index=False)
        print(f"Predictions saved to {opt.output_predictions}")

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix - ' + opt.experiment)
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

def evaluate_model(opt):
    # Load the model
    model_dir = opt.load_model_dir
    model_path = os.path.join(model_dir, 'pytorch_model.bin')
    if not os.path.isfile(model_path):
        print(f"Model file not found at {model_path}")
        return
    # Load the config
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(config_path):
        print(f"Config file not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    label_mapping = config['label_mapping']
    num_labels = len(label_mapping)
    print(f"Config loaded from {config_path}")
    # Reconstruct label encoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_mapping)
    model = ViTClassificationModel(num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Prepare the dataset (same as in training)
    imagedir = './data/' + opt.imagedir
    datasource = opt.dataset.split("-")[0]
    if (datasource == "lamost" or datasource == "slomost"):
        spectradb = pd.read_csv(os.path.join("./data", 'meta-lamost-' + opt.dataset.split("-")[1] + '.csv'))
    else:
        spectradb = pd.read_csv(os.path.join("./data", 'meta-sdss-' + opt.dataset.split("-")[1] + '.csv'))
    for column in spectradb.columns:
        if pd.api.types.is_numeric_dtype(spectradb[column]):
            spectradb[column] = spectradb[column].fillna(0)
        else:
            spectradb[column] = spectradb[column].fillna('null')
    # Clean labels
    spectradb[opt.label_field] = spectradb[opt.label_field].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    spectradb[opt.label_field] = spectradb[opt.label_field].fillna('Unknown')
    debug(opt, f"Dataframe cleaned of missing values and complex characters")
    # Update the DataFrame with filenames and gather available image files
    if (datasource == "lamost"):
        spectradb['filename'] = spectradb.apply(
                lambda row: f'{row["obsdate"].replace("-","")}/{row["planid"]}/spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.png', axis=1
                )
    else:
        spectradb['filename'] = spectradb.apply(
                lambda row: f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.png', axis=1
                )
        spectradb.rename(columns={'subClass': 'subclass'}, inplace=True)
    debug(opt, f"Image filenames added to the dataframe")
    # Check if files exist using vectorized operation and filter the DataFrame
    spectradb['file_exists'] = spectradb['filename'].apply(lambda f: os.path.isfile(os.path.join(imagedir, f)))
    # Keep only rows where the file exists
    spectradb = spectradb[spectradb['file_exists']].drop(columns='file_exists')
    debug(opt, f"Missing filenames dropped")
    image_files = spectradb['filename'].tolist()
    # Get the labels for classification
    labels_list = spectradb[opt.label_field].tolist()
    labels = label_encoder.transform(labels_list)
    debug(opt, f"Found {len(image_files)} image files.")

    # Prepare the dataset as a list of dictionaries
    data = [{"image_path": os.path.join(imagedir, img), "labels": label} for img, label in zip(image_files, labels)]

    # Define the same transform pipeline as in training
    transform_pipeline = T.Compose([
        T.Resize((224, 224)),  # Resize to the size expected by ViT
        T.ToTensor(),          # Convert PIL image to tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    labels = [item['labels'] for item in data]
    images = [item['image_path'] for item in data]

    # Prepare the dataset
    eval_dataset = CustomDataset(images, labels, transform=transform_pipeline)

    # Create DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False)

    # Run inference
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(pixel_values)
            logits = outputs.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            predictions.extend(preds)
            true_labels.extend(batch['labels'].numpy())

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)
    print(f"Mean Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Optionally, save predictions and labels to a CSV
    if opt.output_predictions:
        inverse_label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        preds_labels = [inverse_label_mapping[pred] for pred in predictions]
        true_labels_str = [inverse_label_mapping[label] for label in true_labels]
        results = pd.DataFrame({'image_path': images, 'prediction': preds_labels, 'label': true_labels_str})
        results.to_csv(opt.output_predictions, index=False)
        print(f"Predictions saved to {opt.output_predictions}")

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix - Evaluation')
    plt.savefig('eval_confusion_matrix-' + opt.experiment + '.png')
    plt.close()

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
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model after training')
    parser.add_argument('--pretrained_model_dir', type=str, default=None, help='Path to pretrained model if desired')

    # Evaluation arguments
    parser.add_argument('--do_train', action='store_true', help='Run training')
    parser.add_argument('--do_eval', action='store_true', help='Run evaluation')
    parser.add_argument('--load_model_dir', type=str, default=None, help='Directory containing the saved model to load for evaluation')
    parser.add_argument('--output_predictions', type=str, default=None, help='Path to save predictions CSV file')

    parser.add_argument('--label_field', choices=['class', 'subclass'], default='class', help='Which metadata field to use as label')

    opt = parser.parse_args()
    run(opt)

