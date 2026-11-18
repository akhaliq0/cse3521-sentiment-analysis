
# Amazon Food Review Sentiment Analysis

CSE 3521 Group Project — Ohio State University

## Project Overview

This project uses machine learning to classify Amazon food reviews into three sentiment categories:

- Positive (4–5 stars)
- Neutral (3 stars)
- Negative (1–2 stars)

We use Bag of Words and TF-IDF for text feature extraction and compare machine learning models including Naive Bayes, Logistic Regression, and a Feedforward Neural Network.

## Streamlit Application

The application allows users to enter a food review and receive a predicted sentiment.

The current implementation uses a trained Naive Bayes model with Bag-of-Words features. Additional model comparisons are planned for Milestone 3.

## Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/akhaliq0/cse3521-sentiment-analysis.git
cd cse3521-sentiment-analysis
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal (usually `http://localhost:8501`).

## Project Structure

- `app.py` — Streamlit application
- `models/` — Saved machine learning models
- `data/` — Feature vectors, vectorizers, and dataset files
- `notebook/` — Data processing and model training notebooks
- `results/` — Model evaluation results
- `report/` — Project reports and figures

## Dataset

Amazon Fine Food Reviews:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Our project uses a random sample of 50,000 reviews with seed 43.

## Deployment

Target platform: Streamlit Community Cloud.
