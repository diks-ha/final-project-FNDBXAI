import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re
import joblib
import argparse
import os
import json
from backend.utils import ensure_dir
from torch.optim.lr_scheduler import StepLR

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """Load the dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data: map labels to binary, clean text."""
    # Keep only text and label columns, drop rows with missing values and make an explicit copy
    df = df[['text', 'label']].dropna().copy()

    # Map labels to binary (handle different casing)
    label_mapping = {'REAL': 1, 'FAKE': 0}
    df['label'] = df['label'].astype(str).str.upper().map(label_mapping)

    # Drop any rows that could not be mapped and reset index
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    # Clean text: lowercase, remove punctuation (use pandas vectorized ops)
    df['text'] = df['text'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

    return df.reset_index(drop=True)

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False, pin_memory=False):
    ds = FakeNewsDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory) 

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader, device):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts.extend(d["text"])
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            # Convert tensors to Python types to avoid stacking issues
            predictions.extend(preds.cpu().tolist())
            prediction_probs.extend(outputs.logits.cpu().tolist())
            real_values.extend(labels.cpu().tolist())

    predictions = torch.tensor(predictions)
    prediction_probs = torch.tensor(prediction_probs)
    real_values = torch.tensor(real_values)

    return texts, predictions, prediction_probs, real_values

def main(batch_size=16, max_len=512, epochs=10, debug=False, sample_size=500):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = load_data('data/dataset.csv')

    # Preprocess data
    df = preprocess_data(df)

    # If debugging, sample a small subset to iterate quickly
    if debug:
        print(f"Debug mode: sampling {sample_size} examples from data")
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True).copy()

    # Split data (stratified)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=df_test['label'])

    # Ensure clean copies and reset indices
    df_train = df_train.reset_index(drop=True).copy()
    df_val = df_val.reset_index(drop=True).copy()
    df_test = df_test.reset_index(drop=True).copy()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create data loaders (shuffle for train)
    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_data_loader = create_data_loader(df_val, tokenizer, max_len, batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # Initialize model (set num_labels dynamically)
    num_labels = int(df.label.nunique())
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model = model.to(device)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(df_train.label), y=df_train.label)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Loss function with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Training loop with early stopping
    best_accuracy = 0
    patience = 3
    patience_counter = 0

    try:
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train)
            )

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val)
            )

            print(f'Val loss {val_loss} accuracy {val_acc}')

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model_state.bin')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model state...")
        torch.save(model.state_dict(), 'interrupted_model_state.bin')

    # Load best model (or interrupted one if early stop saved none)
    model_state_path = 'best_model_state.bin' if os.path.exists('best_model_state.bin') else 'interrupted_model_state.bin'
    if os.path.exists(model_state_path):
        model.load_state_dict(torch.load(model_state_path))

    # Evaluate on test set
    y_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader,
        device
    )

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    # Save model and tokenizer
    model.save_pretrained('bert_fake_news_model')
    tokenizer.save_pretrained('bert_fake_news_model')

    # Save metrics for comparison
    ensure_dir('data')
    bert_results = {
        'BERT': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    }

    # If model_comparison.json exists, merge, else write new
    mc_path = 'data/model_comparison.json'
    if os.path.exists(mc_path):
        with open(mc_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(bert_results)
    with open(mc_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)

    print("Model and tokenizer saved to 'bert_fake_news_model' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune BERT on fake news dataset')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a small sample')
    parser.add_argument('--sample-size', type=int, default=500, help='Number of samples to use in debug mode')

    args = parser.parse_args()
    main(batch_size=args.batch_size, max_len=args.max_len, epochs=args.epochs, debug=args.debug, sample_size=args.sample_size)
