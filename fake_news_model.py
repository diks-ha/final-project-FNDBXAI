import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import re

def load_data(file_path):
    """Load the dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data: map labels to binary, clean text."""
    # Handle missing values (if any)
    df = df.dropna()

    # Map labels to binary: REAL -> 1, FAKE -> 0
    label_mapping = {'REAL': 1, 'FAKE': 0}
    df['label'] = df['label'].map(label_mapping)

    # Clean text: lowercase, remove punctuation
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    return df

def extract_features(df):
    """Extract features: TF-IDF on text."""
    print("Sample texts after preprocessing:")
    print(df['text'].head())
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    return X, y, vectorizer

def train_model(X_train, y_train):
    """Train the Logistic Regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

def save_model(model, vectorizer, model_path='fake_news_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    """Save the model and vectorizer."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def main():
    # Load data
    df = load_data('data/dataset.csv')

    # Preprocess data
    df = preprocess_data(df)

    # Extract features
    X, y, vectorizer = extract_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()
