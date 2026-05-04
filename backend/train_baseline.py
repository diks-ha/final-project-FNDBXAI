import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from backend.utils import preprocess_df, save_json, ensure_dir


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def main(dataset_path='data/dataset.csv'):
    df = pd.read_csv(dataset_path)
    df = preprocess_df(df)

    # dataset stats
    stats = {
        'num_samples': int(len(df)),
        'class_counts': df['label'].value_counts().to_dict()
    }
    if 'topic' in df.columns:
        stats['topic_counts'] = df['topic'].value_counts().to_dict()

    ensure_dir('data')
    save_json(stats, 'data/dataset_stats.json')

    # splits
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=df_test['label'])
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train = df_train['text'].tolist()
    y_train = df_train['label'].tolist()
    X_val = df_val['text'].tolist()
    y_val = df_val['label'].tolist()
    X_test = df_test['text'].tolist()
    y_test = df_test['label'].tolist()

    # vectorizer
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    ensure_dir('data/models')
    joblib.dump(vectorizer, 'data/models/tfidf_vectorizer.joblib')

    results = {}

    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    results['LogisticRegression_TFIDF'] = compute_metrics(y_test, y_pred_lr)
    joblib.dump(lr, 'data/models/logistic_regression_tfidf.joblib')

    # Linear SVM
    svm = LinearSVC(class_weight='balanced', max_iter=10000)
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm = svm.predict(X_test_tfidf)
    results['LinearSVM_TFIDF'] = compute_metrics(y_test, y_pred_svm)
    joblib.dump(svm, 'data/models/linear_svm_tfidf.joblib')

    # Save model comparison
    save_json(results, 'data/model_comparison.json')

    print('Baseline training complete. Results saved to data/model_comparison.json')


if __name__ == '__main__':
    main()
