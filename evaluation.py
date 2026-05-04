import pandas as pd
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import re

def load_models():
    """Load both TF-IDF and BERT models"""
    # Load TF-IDF model
    tfidf_model = joblib.load('fake_news_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Load BERT model
    bert_model = BertForSequenceClassification.from_pretrained('bert_fake_news_model')
    bert_tokenizer = BertTokenizer.from_pretrained('bert_fake_news_model')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    return tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device

def preprocess_text(text):
    """Preprocess text for prediction"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_tfidf_batch(texts, model, vectorizer):
    """Make batch predictions using TF-IDF model"""
    processed_texts = [preprocess_text(text) for text in texts]
    text_vectorized = vectorizer.transform(processed_texts)
    predictions = model.predict(text_vectorized)
    probabilities = model.predict_proba(text_vectorized)

    return predictions, probabilities

def predict_bert_batch(texts, model, tokenizer, device, batch_size=16):
    """Make batch predictions using BERT model"""
    predictions = []
    probabilities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        processed_texts = [preprocess_text(text) for text in batch_texts]

        encodings = tokenizer(
            processed_texts,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_probs = probs.cpu().numpy()

            predictions.extend(batch_preds)
            probabilities.extend(batch_probs)

    return np.array(predictions), np.array(probabilities)

def load_test_data():
    """Load test data for evaluation"""
    try:
        df = pd.read_csv('data/dataset.csv')

        # Preprocess
        df = df.dropna()
        label_mapping = {'REAL': 1, 'FAKE': 0}
        df['label'] = df['label'].map(label_mapping)

        # Split to get test set (same split as training)
        _, df_test = train_test_split(df, test_size=0.2, random_state=42)

        return df_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def evaluate_models():
    """Evaluate both models on test data"""
    print("Loading models...")
    tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device = load_models()

    print("Loading test data...")
    df_test = load_test_data()

    if df_test is None:
        print("Could not load test data. Please ensure data/dataset.csv exists.")
        return

    texts = df_test['text'].tolist()
    true_labels = df_test['label'].values

    print("Making predictions with TF-IDF model...")
    tfidf_preds, tfidf_probs = predict_tfidf_batch(texts, tfidf_model, tfidf_vectorizer)

    print("Making predictions with BERT model...")
    bert_preds, bert_probs = predict_bert_batch(texts, bert_model, bert_tokenizer, device)

    # Calculate metrics
    models = {
        'TF-IDF + Logistic Regression': {
            'predictions': tfidf_preds,
            'probabilities': tfidf_probs[:, 1]  # Probability of positive class
        },
        'BERT': {
            'predictions': bert_preds,
            'probabilities': bert_probs[:, 1]  # Probability of positive class
        }
    }

    results = {}

    for model_name, preds_data in models.items():
        predictions = preds_data['predictions']
        probabilities = preds_data['probabilities']

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        auc = roc_auc_score(true_labels, probabilities)

        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc,
            'predictions': predictions,
            'probabilities': probabilities
        }

        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

    return results, true_labels

def plot_comparison(results, true_labels):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: TF-IDF vs BERT', fontsize=16)

    model_names = list(results.keys())
    colors = ['skyblue', 'lightcoral']

    # Metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35

    tfidf_scores = [results['TF-IDF + Logistic Regression'][m] for m in metrics]
    bert_scores = [results['BERT'][m] for m in metrics]

    axes[0, 0].bar(x - width/2, tfidf_scores, width, label='TF-IDF', color=colors[0], alpha=0.8)
    axes[0, 0].bar(x + width/2, bert_scores, width, label='BERT', color=colors[1], alpha=0.8)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Confusion matrices
    for i, model_name in enumerate(model_names):
        cm = confusion_matrix(true_labels, results[model_name]['predictions'])
        row, col = (0, 1) if i == 0 else (1, 0)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        axes[row, col].set_title(f'{model_name} Confusion Matrix')
        axes[row, col].set_ylabel('True Label')
        axes[row, col].set_xlabel('Predicted Label')

    # ROC curves
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for i, model_name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(true_labels, results[model_name]['probabilities'])
        auc_score = results[model_name]['AUC']
        axes[1, 1].plot(fpr, tpr, color=colors[i], alpha=0.8,
                       label=f'{model_name} (AUC = {auc_score:.3f})')

    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def save_results(results, true_labels):
    """Save evaluation results to files"""
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['Accuracy'] for m in results],
        'Precision': [results[m]['Precision'] for m in results],
        'Recall': [results[m]['Recall'] for m in results],
        'F1-Score': [results[m]['F1-Score'] for m in results],
        'AUC': [results[m]['AUC'] for m in results]
    })

    metrics_df.to_csv('figures/model_comparison_metrics.csv', index=False)

    # Save detailed classification reports
    with open('figures/classification_reports.txt', 'w') as f:
        for model_name, data in results.items():
            f.write(f"\n{model_name} Classification Report:\n")
            f.write("-" * 50 + "\n")
            report = classification_report(true_labels, data['predictions'],
                                        target_names=['Fake', 'Real'])
            f.write(report + "\n")

    print("Results saved to figures/ directory")

def main():
    print("Starting model evaluation...")

    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)

    # Evaluate models
    results, true_labels = evaluate_models()

    if results:
        # Create and save comparison plots
        fig = plot_comparison(results, true_labels)
        fig.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save results
        save_results(results, true_labels)

        print("\nEvaluation completed successfully!")
        print("Results saved in figures/ directory")
    else:
        print("Evaluation failed.")

if __name__ == "__main__":
    main()
