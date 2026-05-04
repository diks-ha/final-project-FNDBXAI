import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Set page config
st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="📰",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
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

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('data/dataset.csv')
        return df
    except:
        return None

def preprocess_text(text):
    """Preprocess text for prediction"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_tfidf(text, model, vectorizer):
    """Make prediction using TF-IDF model"""
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]

    return prediction, probability

def predict_bert(text, model, tokenizer, device):
    """Make prediction using BERT model"""
    processed_text = preprocess_text(text)

    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        probability = probs[0][prediction].item()

    return prediction, probability

def main():
    st.title("📰 Fake News Detection Dashboard")
    st.markdown("---")

    # Load models
    try:
        tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure the models are trained and saved first.")
        return

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Prediction", "Model Comparison", "Data Insights"])

    if page == "Prediction":
        st.header("🔍 Make a Prediction")

        # Text input
        user_input = st.text_area(
            "Enter news text to analyze:",
            height=150,
            placeholder="Paste your news article text here..."
        )

        if st.button("Analyze Text", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    # TF-IDF Prediction
                    tfidf_pred, tfidf_prob = predict_tfidf(user_input, tfidf_model, tfidf_vectorizer)

                    # BERT Prediction
                    bert_pred, bert_prob = predict_bert(user_input, bert_model, bert_tokenizer, device)

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 TF-IDF Model Results")
                        if tfidf_pred == 1:
                            st.success("REAL NEWS")
                        else:
                            st.error("FAKE NEWS")
                        st.metric("Confidence", f"{tfidf_prob[tfidf_pred]:.2%}")

                    with col2:
                        st.subheader("🤖 BERT Model Results")
                        if bert_pred == 1:
                            st.success("REAL NEWS")
                        else:
                            st.error("FAKE NEWS")
                        st.metric("Confidence", f"{bert_prob:.2%}")

                    # Consensus
                    if tfidf_pred == bert_pred:
                        st.success("✅ Both models agree on the prediction!")
                    else:
                        st.warning("⚠️ Models disagree. Consider manual verification.")

            else:
                st.warning("Please enter some text to analyze.")

    elif page == "Model Comparison":
        st.header("⚖️ Model Comparison")

        # Load sample data for comparison
        df = load_sample_data()

        if df is not None:
            st.subheader("Sample Data Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Samples", len(df))

            with col2:
                real_count = (df['label'] == 'REAL').sum()
                st.metric("Real News", real_count)

            with col3:
                fake_count = (df['label'] == 'FAKE').sum()
                st.metric("Fake News", fake_count)

            # Distribution chart
            fig, ax = plt.subplots()
            df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title("News Distribution")
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Model Performance Metrics")

            # Placeholder for actual metrics (would need test data)
            metrics_data = {
                'Model': ['TF-IDF + Logistic Regression', 'BERT'],
                'Accuracy': [0.85, 0.92],  # Replace with actual values
                'Precision': [0.83, 0.91],
                'Recall': [0.87, 0.93],
                'F1-Score': [0.85, 0.92]
            }

            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)

        else:
            st.warning("Sample data not found. Please ensure data/dataset.csv exists.")

    elif page == "Data Insights":
        st.header("📈 Data Insights")

        df = load_sample_data()

        if df is not None:
            # Text length analysis
            df['text_length'] = df['text'].str.len()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Text length distribution
            sns.histplot(data=df, x='text_length', hue='label', ax=ax1, alpha=0.7)
            ax1.set_title("Text Length Distribution")
            ax1.set_xlabel("Text Length")

            # Box plot
            sns.boxplot(data=df, x='label', y='text_length', ax=ax2)
            ax2.set_title("Text Length by Category")
            ax2.set_ylabel("Text Length")

            st.pyplot(fig)

            # Word count analysis
            df['word_count'] = df['text'].str.split().str.len()

            st.subheader("Word Count Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Avg Words (Real)", f"{df[df['label']=='REAL']['word_count'].mean():.1f}")

            with col2:
                st.metric("Avg Words (Fake)", f"{df[df['label']=='FAKE']['word_count'].mean():.1f}")

        else:
            st.warning("Sample data not found. Please ensure data/dataset.csv exists.")

if __name__ == "__main__":
    main()
