"""Streamlit interface for CSE 3521 Amazon Food Review sentiment analysis."""

from pathlib import Path
import html
import re

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "nb_bow_model.joblib"
VECTORIZERS_PATH = ROOT / "data" / "vectorizers.joblib"


def clean_text(text: str) -> str:
    """Match preprocessing from the team's M2 data engineering notebook."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    return re.sub(r"\\s+", " ", text).strip()


@st.cache_resource
def load_artifacts():
    """Load the trained BoW Naive Bayes model and its fitted vectorizer."""
    model = joblib.load(MODEL_PATH)
    saved_vectorizers = joblib.load(VECTORIZERS_PATH)

    if isinstance(saved_vectorizers, dict):
        matches = [
            value
            for key, value in saved_vectorizers.items()
            if "bow" in str(key).lower() or "count" in str(key).lower()
        ]
        if not matches:
            raise ValueError(
                "No Bag-of-Words vectorizer found in data/vectorizers.joblib."
            )
        vectorizer = matches[0]
    else:
        vectorizer = saved_vectorizers

    if not hasattr(vectorizer, "transform"):
        raise TypeError("Saved BoW vectorizer has no transform() method.")
    if not hasattr(model, "predict"):
        raise TypeError("Saved model has no predict() method.")

    return model, vectorizer


def predict_review(review: str, model, vectorizer):
    cleaned_review = clean_text(review)
    features = vectorizer.transform([cleaned_review])

    if (
        hasattr(model, "n_features_in_")
        and features.shape[1] != model.n_features_in_
    ):
        raise ValueError(
            "The BoW vectorizer and trained model expect different feature counts."
        )

    prediction = str(model.predict(features)[0])

    probabilities = None
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        scores = model.predict_proba(features)[0]
        probabilities = {
            str(label): float(score)
            for label, score in zip(model.classes_, scores)
        }

    return prediction, probabilities


st.set_page_config(
    page_title="Amazon Food Review Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered",
)

st.title("Amazon Food Review Sentiment Analyzer")
st.write(
    "Enter a food product review to see whether our trained model "
    "predicts **Positive**, **Neutral**, or **Negative** sentiment."
)

analyze_tab, about_tab = st.tabs(["Analyze a Review", "About the Project"])

with analyze_tab:
    with st.form("review_form"):
        review = st.text_area(
            "Food review",
            placeholder="Example: The snacks arrived fresh and tasted great.",
            height=160,
        )
        submitted = st.form_submit_button("Analyze Sentiment")

    if submitted:
        if not review.strip():
            st.warning("Enter a review before analyzing it.")
        else:
            try:
                model, vectorizer = load_artifacts()
                label, probabilities = predict_review(
                    review, model, vectorizer
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "Check that models/nb_bow_model.joblib and "
                    "data/vectorizers.joblib are present and compatible."
                )
            else:
                st.subheader(f"Predicted Sentiment: {label}")

                if probabilities:
                    probability_table = pd.DataFrame(
                        {
                            "Sentiment": list(probabilities.keys()),
                            "Estimated probability": list(probabilities.values()),
                        }
                    ).set_index("Sentiment")
                    st.bar_chart(probability_table)
                    st.caption(
                        "Probabilities are estimates from the trained model, "
                        "not a guarantee that the prediction is correct."
                    )

with about_tab:
    st.subheader("How it works")
    st.write(
        "The app cleans the entered review, converts the text into "
        "Bag-of-Words features using our saved fitted vectorizer, "
        "and runs a trained Multinomial Naive Bayes classifier."
    )
    st.write(
        "**Dataset:** 50,000 Amazon Fine Food Reviews. "
        "**Labels:** Negative (1–2 stars), Neutral (3 stars), "
        "Positive (4–5 stars)."
    )
    st.write(
        "**Milestone 2 baseline:** Naive Bayes with Bag of Words "
        "achieved 82.6% held-out test accuracy and 0.596 macro F1."
    )
    st.caption(
        "Additional model comparisons will be integrated in Milestone 3."
    )
