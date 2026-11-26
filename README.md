
# Amazon Food Review Sentiment Analysis

CSE 3521 Group Project — Ohio State University

## Project Overview

This project uses machine learning to classify Amazon food reviews into three sentiment categories:

- Positive (4–5 stars)
- Neutral (3 stars)
- Negative (1–2 stars)

We use Bag of Words and TF-IDF for text feature extraction and compare machine learning models including Naive Bayes, Logistic Regression, and a Feedforward Neural Network.


## Streamlit Application

The application provides three tabs:

- **Analyze a Review:** Compare live sentiment predictions and class probabilities from three trained classifiers.
- **Model Performance:** Evaluate accuracy, macro precision, macro recall, and macro F1-score using the held-out test dataset.
- **About the Project:** Learn about the dataset and machine learning approach.

### Integrated Models

- Multinomial Naive Bayes (Bag of Words)
- Logistic Regression (Bag of Words)
- Logistic Regression (TF-IDF)

### Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

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
