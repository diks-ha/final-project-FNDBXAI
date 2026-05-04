import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import argparse
import sys

def load_models():
    """Load both TF-IDF and BERT models"""
    try:
        # Load TF-IDF model
        tfidf_model = joblib.load('fake_news_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("✅ TF-IDF model loaded successfully")

        # Load BERT model
        bert_model = BertForSequenceClassification.from_pretrained('bert_fake_news_model')
        bert_tokenizer = BertTokenizer.from_pretrained('bert_fake_news_model')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)
        bert_model.eval()
        print("✅ BERT model loaded successfully")
        print(f"Using device: {device}")

        return tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("Please ensure models are trained and saved first.")
        return None, None, None, None, None

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

def print_prediction_results(text, tfidf_pred, tfidf_prob, bert_pred, bert_prob):
    """Print formatted prediction results"""
    print("\n" + "="*60)
    print("📰 FAKE NEWS DETECTION RESULTS")
    print("="*60)

    print(f"\n📝 Input Text (first 100 chars):")
    print(f"'{text[:100]}{'...' if len(text) > 100 else ''}'")
    print(f"📏 Text Length: {len(text)} characters")

    print(f"\n🤖 Model Predictions:")
    print("-" * 40)

    # TF-IDF Results
    tfidf_label = "REAL" if tfidf_pred == 1 else "FAKE"
    tfidf_color = "🟢" if tfidf_pred == 1 else "🔴"
    print(f"📊 TF-IDF + Logistic Regression:")
    print(f"   {tfidf_color} Prediction: {tfidf_label}")
    print(".2%")

    # BERT Results
    bert_label = "REAL" if bert_pred == 1 else "FAKE"
    bert_color = "🟢" if bert_pred == 1 else "🔴"
    print(f"\n🧠 BERT Transformer:")
    print(f"   {bert_color} Prediction: {bert_label}")
    print(".2%")

    # Consensus
    print(f"\n⚖️  Consensus:")
    if tfidf_pred == bert_pred:
        consensus_label = "REAL" if tfidf_pred == 1 else "FAKE"
        consensus_color = "🟢" if tfidf_pred == 1 else "🔴"
        max_confidence = max(tfidf_prob, bert_prob)
        print(f"   {consensus_color} Both models agree: {consensus_label}")
        print(".2%")
    else:
        print("   ⚠️  Models disagree - Manual review recommended")
    print("\n" + "="*60)

def interactive_mode(tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device):
    """Run interactive prediction mode"""
    print("\n" + "="*60)
    print("📰 FAKE NEWS DETECTOR - Interactive Mode")
    print("="*60)
    print("Enter news text to analyze (or 'quit' to exit):")
    print("-" * 60)

    while True:
        try:
            text = input("\n🔍 Enter text: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not text:
                print("⚠️  Please enter some text.")
                continue

            # Make predictions
            tfidf_pred, tfidf_prob = predict_tfidf(text, tfidf_model, tfidf_vectorizer)
            bert_pred, bert_prob = predict_bert(text, bert_model, bert_tokenizer, device)

            # Display results
            print_prediction_results(text, tfidf_pred, tfidf_prob, bert_pred, bert_prob)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing text: {str(e)}")

def file_mode(file_path, tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device):
    """Process text from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        if not text:
            print("❌ File is empty.")
            return

        print(f"📖 Processing text from file: {file_path}")

        # Make predictions
        tfidf_pred, tfidf_prob = predict_tfidf(text, tfidf_model, tfidf_vectorizer)
        bert_pred, bert_prob = predict_bert(text, bert_model, bert_tokenizer, device)

        # Display results
        print_prediction_results(text, tfidf_pred, tfidf_prob, bert_pred, bert_prob)

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection Demo')
    parser.add_argument('--text', '-t', type=str, help='Text to analyze')
    parser.add_argument('--file', '-f', type=str, help='File containing text to analyze')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')

    args = parser.parse_args()

    # Load models
    models = load_models()
    if not all(models):
        return

    tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device = models

    # Determine mode
    if args.interactive:
        interactive_mode(tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device)
    elif args.file:
        file_mode(args.file, tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device)
    elif args.text:
        # Make predictions
        tfidf_pred, tfidf_prob = predict_tfidf(args.text, tfidf_model, tfidf_vectorizer)
        bert_pred, bert_prob = predict_bert(args.text, bert_model, bert_tokenizer, device)

        # Display results
        print_prediction_results(args.text, tfidf_pred, tfidf_prob, bert_pred, bert_prob)
    else:
        print("🤔 No input provided. Use --help for usage instructions.")
        print("\nQuick examples:")
        print("  python demo.py --text \"Your news text here\"")
        print("  python demo.py --file news_article.txt")
        print("  python demo.py --interactive")

if __name__ == "__main__":
    main()
