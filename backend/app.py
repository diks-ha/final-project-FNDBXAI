from flask import Flask, request, jsonify
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models at startup
tfidf_model = None
tfidf_vectorizer = None
bert_model = None
bert_tokenizer = None
device = None

def load_models():
    """Load both TF-IDF and BERT models"""
    global tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer, device

    try:
        # Load TF-IDF model
        tfidf_model = joblib.load('./fake_news_model.pkl')
        tfidf_vectorizer = joblib.load('./tfidf_vectorizer.pkl')
        logger.info("TF-IDF model loaded successfully")

        # Load BERT model
        bert_model = BertForSequenceClassification.from_pretrained('./bert_fake_news_model')
        bert_tokenizer = BertTokenizer.from_pretrained('./bert_fake_news_model')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)
        bert_model.eval()
        logger.info("BERT model loaded successfully")

        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_text(text):
    """Preprocess text for prediction"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/', methods=['GET'])
def root():
    """API information endpoint"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fake News Detection API</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .version {
                text-align: center;
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }
            .endpoints {
                margin: 30px 0;
            }
            .endpoint {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #4CAF50;
            }
            .method {
                font-weight: bold;
                color: #4CAF50;
                margin-right: 10px;
            }
            .example {
                margin: 30px 0;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 20px;
            }
            .code {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 5px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                margin: 10px 0;
                overflow-x: auto;
            }
            .status {
                text-align: center;
                margin: 20px 0;
                padding: 10px;
                border-radius: 5px;
                background: rgba(76, 175, 80, 0.2);
                border: 1px solid rgba(76, 175, 80, 0.3);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📰 Fake News Detection API</h1>
            <div class="version">Version 1.0</div>

            <div class="status">
                ✅ API is running and ready to use
            </div>

            <div class="endpoints">
                <h2>📋 Available Endpoints</h2>
                <div class="endpoint">
                    <span class="method">GET</span> / - API information (this page)
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> /health - Health check
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> /predict - Single prediction
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> /batch_predict - Batch predictions
                </div>
            </div>

            <div class="example">
                <h2>🚀 Quick Start Example</h2>
                <p>Use this curl command to test the API:</p>
                <div class="code">
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Your news article text here"}'
                </div>

                <p>Expected response:</p>
                <div class="code">
{
  "text_length": 123,
  "processed_text_length": 115,
  "models": {
    "tfidf": {
      "prediction": "REAL",
      "confidence": 0.9876,
      "prediction_numeric": 1
    },
    "bert": {
      "prediction": "REAL",
      "confidence": 0.9456,
      "prediction_numeric": 1
    }
  },
  "consensus": {
    "prediction": "REAL",
    "confidence": 0.9876
  }
}
                </div>
            </div>

            <div class="example">
                <h2>📊 Dashboard</h2>
                <p>For interactive predictions and visualizations, run:</p>
                <div class="code">streamlit run dashboard.py</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': all([tfidf_model, tfidf_vectorizer, bert_model, bert_tokenizer])
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on news text"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please include "text" field in JSON payload.'
            }), 400

        text = data['text']
        if not text.strip():
            return jsonify({
                'error': 'Empty text provided.'
            }), 400

        results = {}

        # TF-IDF Prediction
        if tfidf_model and tfidf_vectorizer:
            processed_text = preprocess_text(text)
            text_vectorized = tfidf_vectorizer.transform([processed_text])
            tfidf_pred = int(tfidf_model.predict(text_vectorized)[0])
            tfidf_prob = float(tfidf_model.predict_proba(text_vectorized)[0][tfidf_pred])

            results['tfidf'] = {
                'prediction': 'REAL' if tfidf_pred == 1 else 'FAKE',
                'confidence': round(tfidf_prob, 4),
                'prediction_numeric': tfidf_pred
            }

        # BERT Prediction
        if bert_model and bert_tokenizer:
            processed_text = preprocess_text(text)

            encoding = bert_tokenizer.encode_plus(
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
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                bert_pred = int(torch.argmax(logits, dim=1).item())
                bert_prob = float(probs[0][bert_pred].item())

            results['bert'] = {
                'prediction': 'REAL' if bert_pred == 1 else 'FAKE',
                'confidence': round(bert_prob, 4),
                'prediction_numeric': bert_pred
            }

        if not results:
            return jsonify({
                'error': 'Models not loaded properly.'
            }), 500

        # Consensus prediction
        if 'tfidf' in results and 'bert' in results:
            tfidf_pred = results['tfidf']['prediction_numeric']
            bert_pred = results['bert']['prediction_numeric']

            if tfidf_pred == bert_pred:
                consensus = results['tfidf']['prediction']
                confidence = max(results['tfidf']['confidence'], results['bert']['confidence'])
            else:
                consensus = 'UNCERTAIN'
                confidence = 0.5
        else:
            consensus = results[list(results.keys())[0]]['prediction']
            confidence = results[list(results.keys())[0]]['confidence']

        response = {
            'text_length': len(text),
            'processed_text_length': len(processed_text),
            'models': results,
            'consensus': {
                'prediction': consensus,
                'confidence': confidence
            }
        }

        logger.info(f"Prediction made for text of length {len(text)}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during prediction.'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make batch predictions on multiple texts"""
    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({
                'error': 'No texts provided. Please include "texts" field (array) in JSON payload.'
            }), 400

        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'error': 'Texts must be a non-empty array.'
            }), 400

        if len(texts) > 100:  # Limit batch size
            return jsonify({
                'error': 'Maximum 100 texts allowed in batch prediction.'
            }), 400

        results = []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                results.append({
                    'index': i,
                    'error': 'Text must be a string.'
                })
                continue

            try:
                # Single prediction logic (reuse from predict endpoint)
                single_result = predict_single(text)
                results.append({
                    'index': i,
                    'result': single_result
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })

        return jsonify({
            'batch_size': len(texts),
            'results': results
        })

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during batch prediction.'
        }), 500

def predict_single(text):
    """Helper function for single prediction (used in batch)"""
    processed_text = preprocess_text(text)
    results = {}

    # TF-IDF Prediction
    if tfidf_model and tfidf_vectorizer:
        text_vectorized = tfidf_vectorizer.transform([processed_text])
        tfidf_pred = int(tfidf_model.predict(text_vectorized)[0])
        tfidf_prob = float(tfidf_model.predict_proba(text_vectorized)[0][tfidf_pred])

        results['tfidf'] = {
            'prediction': 'REAL' if tfidf_pred == 1 else 'FAKE',
            'confidence': round(tfidf_prob, 4)
        }

    # BERT Prediction
    if bert_model and bert_tokenizer:
        encoding = bert_tokenizer.encode_plus(
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
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            bert_pred = int(torch.argmax(logits, dim=1).item())
            bert_prob = float(probs[0][bert_pred].item())

        results['bert'] = {
            'prediction': 'REAL' if bert_pred == 1 else 'FAKE',
            'confidence': round(bert_prob, 4)
        }

    return results

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if load_models():
        logger.info("Starting Flask API server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load models. Exiting.")
        exit(1)
