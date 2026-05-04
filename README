LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.
=====================================================================
Description of the TSV format:

Column 1: the ID of the statement ([ID].json).
Column 2: the label.
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).

Note that we do not provide the full-text verdict report in this current version of the dataset,
but you can use the following command to access the full verdict report and links to the source documents:
wget http://www.politifact.com//api/v/2/statement/[ID]/?format=json

======================================================================
The original sources retain the copyright of the data.

Note that there are absolutely no guarantees with this data,
and we provide this dataset "as is",
but you are welcome to report the issues of the preliminary version
of this data.

You are allowed to use this dataset for research purposes only.

For more question about the dataset, please contact:
William Wang, william@cs.ucsb.edu

v1.0 04/23/2017

======================================================================
RUNNING INSTRUCTIONS
======================================================================

This project provides multiple ways to interact with the fake news detection models:

## Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models (if not already trained):**
   ```bash
   # Train baseline TF-IDF + Logistic Regression model
   python fake_news_model.py

   # Train BERT model
   python bert_model.py
   ```

## Usage Options

### 1. Command-Line Demo
Run predictions from the command line:

```bash
# Interactive mode
python demo.py --interactive

# Analyze specific text
python demo.py --text "Your news article text here"

# Analyze text from file
python demo.py --file path/to/news_article.txt
```

### 2. Web Dashboard
Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501` and provides:
- Text prediction interface
- Model comparison metrics
- Data insights and visualizations

### 3. REST API
Start the Flask API server:

```bash
python backend/app.py
```

The API will be available at `http://localhost:5000` with endpoints:
- `GET /` - API documentation
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions

Example API usage:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'
```

## Project Structure

- `fake_news_model.py` - Baseline model training (TF-IDF + Logistic Regression)
- `bert_model.py` - BERT-based model training
- `demo.py` - Command-line prediction tool
- `dashboard.py` - Streamlit web dashboard
- `backend/app.py` - Flask REST API
- `evaluation.py` - Model evaluation and comparison
- `visualization.py` - Data visualization scripts
- `data/` - Dataset and model artifacts
- `figures/` - Generated plots and charts

