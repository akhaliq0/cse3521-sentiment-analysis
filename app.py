"""CSE 3521 sentiment analysis — multi-model Streamlit prototype."""
import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
SPECS = [
    ('Naive Bayes (BoW)', 'nb_bow_model.joblib', 'bow'),
    ('Logistic Regression (BoW)', 'lr_bow_model.joblib', 'bow'),
    ('Logistic Regression (TF-IDF)', 'lr_tfidf_model.joblib', 'tfidf'),
]
LABELS = ['Negative', 'Neutral', 'Positive']


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text).lower()
    return re.sub(r'\s+', ' ', text).strip()


@st.cache_resource
def load_model(filename):
    return joblib.load(MODELS / filename)


@st.cache_resource
def load_vectorizer(kind):
    saved = joblib.load(DATA / 'vectorizers.joblib')
    values = list(saved.values()) if isinstance(saved, dict) else (
        list(saved) if isinstance(saved, (list, tuple)) else [saved]
    )
    if kind == 'bow':
        matches = [
            v for v in values
            if isinstance(v, CountVectorizer)
            and not isinstance(v, TfidfVectorizer)
        ]
    elif kind == 'tfidf':
        matches = [v for v in values if isinstance(v, TfidfVectorizer)]
    else:
        raise ValueError(f'Unknown vectorizer type: {kind}')
    if len(matches) != 1:
        raise ValueError(f'Expected one fitted {kind} vectorizer; found {len(matches)}.')
    return matches[0]


def predict(review, filename, kind):
    model = load_model(filename)
    features = load_vectorizer(kind).transform([clean_text(review)])
    if features.shape[1] != model.n_features_in_:
        raise ValueError('Model and vectorizer feature counts do not match.')
    if features.nnz == 0:
        raise ValueError('No recognized vocabulary words in this review.')
    label = str(model.predict(features)[0])
    probabilities = dict(zip(map(str, model.classes_), model.predict_proba(features)[0]))
    return label, probabilities


@st.cache_data
def evaluate(filename, kind):
    model = load_model(filename)
    X = load_npz(DATA / f'X_test_{kind}.npz')
    labels = pd.read_csv(DATA / 'y_test.csv')
    if 'Sentiment' in labels.columns:
        y = labels['Sentiment']
    elif 'sentiment' in labels.columns:
        y = labels['sentiment']
    elif labels.shape[1] == 1:
        y = labels.iloc[:, 0]
    else:
        raise ValueError(f'Cannot identify label column: {list(labels.columns)}')
    if len(y) != X.shape[0] or set(y.unique()) != set(LABELS):
        raise ValueError('Test labels do not match the expected rows or classes.')
    preds = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(
        y, preds, labels=LABELS, average='macro', zero_division=0
    )
    return {'Accuracy': accuracy_score(y, preds), 'Macro precision': p,
            'Macro recall': r, 'Macro F1': f}


st.set_page_config(page_title='Food Review Sentiment Analyzer', page_icon='🍽️', layout='wide')
st.title('Amazon Food Review Sentiment Analyzer')
st.write('Compare machine learning predictions for a food product review.')
available = [spec for spec in SPECS if (MODELS / spec[1]).is_file()]
st.caption(f'{len(available)} of {len(SPECS)} configured model files available')
analyze, performance, about = st.tabs(['Analyze a Review', 'Model Performance', 'About the Project'])

with analyze:
    with st.form('review_form'):
        review = st.text_area('Food review', height=150,
                              placeholder='These snacks taste great and arrived fresh.')
        submitted = st.form_submit_button('Analyze Sentiment', type='primary')
    if submitted:
        if not clean_text(review):
            st.warning('Please enter a review.')
        else:
            for column, (name, filename, kind) in zip(st.columns(len(SPECS)), SPECS):
                with column:
                    st.subheader(name)
                    if not (MODELS / filename).is_file():
                        st.info('Model not available yet.')
                        continue
                    try:
                        label, probabilities = predict(review, filename, kind)
                        if label == 'Positive':
                            st.success(label)
                        elif label == 'Negative':
                            st.error(label)
                        else:
                            st.info(label)
                        st.bar_chart(pd.DataFrame({'Estimated probability': probabilities}).reindex(LABELS))
                    except Exception as exc:
                        st.warning(f'Prediction unavailable: {exc}')
            st.caption('Model probabilities are estimates, not guarantees of correctness.')

with performance:
    st.subheader('Held-out test set performance')
    st.write('Evaluate the saved classifiers using the team’s existing test features and labels. Precision, recall and F1 are macro-averaged across the three classes.')
    if st.button('Evaluate saved models'):
        rows = []
        for name, filename, kind in available:
            try:
                rows.append({'Model': name, **evaluate(filename, kind)})
            except Exception as exc:
                st.warning(f'{name}: {exc}')
        if rows:
            table = pd.DataFrame(rows).set_index('Model')
            st.dataframe(table.style.format('{:.3f}'), use_container_width=True)
            st.bar_chart(table[['Accuracy', 'Macro F1']], stack=False)
    st.caption('Training time and neural network results will be added when verified.')

with about:
    st.subheader('Dataset and methodology')
    st.write('50,000 Amazon Fine Food Reviews, sampled with seed 43. The team used a stratified 80/20 train-test split.')
    st.write('Negative: 1–2 stars · Neutral: 3 stars · Positive: 4–5 stars')
    st.write('Reviews are cleaned, transformed with the original fitted Bag-of-Words or TF-IDF vectorizer, and passed to the corresponding trained classifier. No retraining is performed by the app.')
    st.write('Naive Bayes and Logistic Regression are classical classifiers. A feedforward neural network is planned for the project; it is not connected in this version.')
