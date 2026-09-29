import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Performance & Sentiment Analysis", page_icon="🎓", layout="centered")

STUDENT_CSV = "student_performance_clean.csv"
REVIEWS_CSV = "sentiment_reviews.csv"
FEATURES = ["study_hours_per_day", "attendance_percent", "sleep_hours", "previous_score", "extracurricular_encoded"]


@st.cache_resource
def train_regression():
    df = pd.read_csv(STUDENT_CSV)
    df["extracurricular_encoded"] = df["extracurricular"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=FEATURES + ["final_score"])
    X, y = df[FEATURES], df["final_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "R²": r2_score(y_te, pred),
        "RMSE": mean_squared_error(y_te, pred) ** 0.5,
        "MAE": mean_absolute_error(y_te, pred),
    }
    return model, metrics, df


@st.cache_resource
def train_sentiment():
    df = pd.read_csv(REVIEWS_CSV).dropna()
    # Find the text column (longest average text) and the label column (the other one)
    text_col = max(df.select_dtypes(include="object").columns, key=lambda c: df[c].astype(str).str.len().mean())
    label_col = [c for c in df.columns if c != text_col][-1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[text_col].astype(str), df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )
    model = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    return model, model.score(X_te, y_te)


def label_name(v):
    s = str(v).strip().lower()
    if s in ("1", "positive", "pos"):
        return "Positive 😊"
    if s in ("0", "negative", "neg"):
        return "Negative 😞"
    return str(v)


st.title("🎓 Student Performance & Sentiment Analysis")
st.caption("AI/ML internship project at Vyrex Technology, by Sangamithra J")

tab1, tab2, tab3 = st.tabs(["Score predictor", "Sentiment checker", "About the project"])

with tab1:
    st.subheader("Predict a student's final exam score")
    reg, m, data = train_regression()
    c1, c2 = st.columns(2)
    study = c1.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
    attend = c1.slider("Attendance (%)", 0, 100, 80)
    sleep = c2.slider("Sleep hours", 3.0, 12.0, 7.0, 0.5)
    prev = c2.slider("Previous score", 0, 100, 60)
    extra = st.radio("Extracurricular activities", ["Yes", "No"], horizontal=True)
    if st.button("Predict score", type="primary"):
        row = pd.DataFrame([[study, attend, sleep, prev, 1 if extra == "Yes" else 0]], columns=FEATURES)
        score = float(reg.predict(row)[0])
        st.success(f"Predicted final score: **{score:.1f}**")
        st.caption(f"Typical error on unseen data is about ±{m['MAE']:.1f} points (MAE).")
    st.markdown(f"**Model performance (20% test set):** R² = {m['R²']:.2f} | RMSE = {m['RMSE']:.2f} | MAE = {m['MAE']:.2f}")

with tab2:
    st.subheader("Is this review positive or negative?")
    clf, acc = train_sentiment()
    text = st.text_area("Type or paste a review", placeholder="The product quality is great and delivery was fast.")
    if st.button("Analyze sentiment", type="primary"):
        if text.strip():
            probs = clf.predict_proba([text])[0]
            best = probs.argmax()
            st.success(f"{label_name(clf.classes_[best])}  (confidence {probs[best]*100:.0f}%)")
        else:
            st.warning("Please enter some text first.")
    st.info(
        f"This classifier scored {acc*100:.0f}% on its test set, but it was trained on a small, simple dataset. "
        "Treat it as a learning demo, not a production-grade model."
    )

with tab3:
    st.subheader("What I built")
    st.markdown(
        "- **Python fundamentals:** prime numbers, word frequency, list statistics, duplicate removal, Fibonacci\n"
        "- **Data cleaning and EDA:** missing values, duplicates, correlation heatmap, score distribution\n"
        "- **Linear regression:** predicts final exam score from study habits, attendance, sleep, and past scores\n"
        "- **Sentiment classifier:** TF-IDF + Logistic Regression for positive vs. negative reviews"
    )
    st.markdown("**Key finding:** study hours per day is the strongest driver of final score, followed by attendance and previous score.")
    for f, cap in [
        ("correlation_heatmap.png", "Correlation heatmap"),
        ("final_score_distribution.png", "Final score distribution"),
        ("predicted_vs_actual.png", "Predicted vs. actual scores"),
        ("confusion_matrix.png", "Sentiment confusion matrix"),
    ]:
        if os.path.exists(f):
            st.image(f, caption=cap)
    st.markdown("[View the code on GitHub](https://github.com/jsangamithra/Student-Performance-Sentiment-Analysis-AI-ML-Internship-Project-Vyrex-Technology-)")
