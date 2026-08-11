# Student-Performance-Sentiment-Analysis-AI-ML-Internship-Project-Vyrex-Technology-
AI/ML internship project at Vyrex Technology — Python fundamentals, data cleaning &amp; EDA, a linear regression model predicting student scores, and a TF-IDF + Logistic Regression sentiment classifier.
# Student Performance & Sentiment Analysis — AI/ML Internship Project

**Internship Project — Vyrex Technology**

## Description

This repository contains the deliverables from my AI/ML internship at **Vyrex Technology**. The project is built in four progressive stages that together demonstrate the core AI/ML development workflow — from fundamental programming logic to a complete supervised machine learning pipeline covering both regression and classification.

The project uses a student performance dataset and a product review dataset to answer two practical questions:
1. **Can we predict a student's final exam score** based on their study habits, attendance, sleep, and past performance?
2. **Can we automatically classify a piece of text (a review) as positive or negative sentiment**, without a human reading it?

Along the way, the project also demonstrates strong Python fundamentals and a full data cleaning / exploratory data analysis (EDA) pipeline, which are the essential building blocks of any real-world ML system.

## Objective / Why This Project

Machine learning models are only as good as the pipeline behind them. This project was designed to prove hands-on competency across the full pipeline an ML/AI role actually requires:
- Writing clean, correct, well-documented Python code
- Handling messy, real-world data (missing values, duplicates) responsibly
- Exploring data visually to form hypotheses before modeling
- Building, training, and rigorously evaluating a regression model
- Building, training, and rigorously evaluating a text classification model
- Interpreting model results honestly, including their limitations

## Project Structure

This project is organized into four tasks:

### Task 1: Python Fundamentals
Five standalone functions covering prime number generation, word frequency counting, list statistics, duplicate removal, and Fibonacci sequence generation — demonstrating core programming logic without relying on built-in shortcut libraries.

### Task 2: Data Cleaning & Exploratory Data Analysis (EDA)
Loads the raw `student_performance.csv` dataset (305 rows), inspects it for missing values and duplicates, cleans it (median imputation for missing numeric values, duplicate removal), and produces summary statistics and visualizations (correlation heatmap, score distribution). Outputs a clean dataset (`student_performance_clean.csv`, 300 rows) for downstream modeling.

### Task 3: Linear Regression — Predicting Final Exam Scores
Trains a Linear Regression model to predict `final_score` using study hours, attendance, sleep hours, previous score, and extracurricular participation as features. Evaluated on a held-out test set using RMSE, MAE, and R².

**Results:** R² = 0.651 | RMSE = 5.807 | MAE = 4.596
Study hours per day was the strongest predictor of final score, followed by attendance percentage and previous score.

### Task 4: Sentiment Classification — Positive vs Negative Reviews
Trains a Logistic Regression classifier on TF-IDF vectorized text (`sentiment_reviews.csv`, 392 labeled reviews) to classify reviews as positive or negative. Evaluated using accuracy, precision, recall, F1-score, and a confusion matrix, then tested on brand-new unseen example sentences.

**Results:** Accuracy = 1.00 | Precision = 1.00 | Recall = 1.00 | F1 = 1.00
*Note: the near-perfect score reflects the size and relative simplicity of this dataset rather than claiming a production-grade, flawless real-world sentiment classifier. Predictions on brand-new unseen sentences showed realistic, sub-1.0 confidence scores, which supports genuine generalization.*

## Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (LinearRegression, LogisticRegression, TfidfVectorizer, StandardScaler, train_test_split)
- **Environment:** Google Colab / Jupyter Notebook

## Key Outcomes

- Built and evaluated a working regression model that explains ~65% of the variance in student final scores, with study hours as the single biggest driver of performance.
- Built and evaluated a working text classification model that reliably distinguishes positive from negative sentiment, including on entirely new, unseen text.
- Delivered a complete, reproducible pipeline: raw data → cleaned data → exploratory visualizations → trained models → evaluation metrics → prediction plots.

## Files to Upload to This Repository

| File | Description |
|---|---|
| `README.md` | This file |
| `vyrex_intern.ipynb` | Main notebook containing all four tasks (Python fundamentals, EDA, regression, classification) |
| `student_performance.csv` | Raw student performance dataset (input to Task 2) |
| `student_performance_clean.csv` | Cleaned dataset generated by Task 2, used in Task 3 |
| `sentiment_reviews.csv` | Labeled review dataset used in Task 4 |
| `correlation_heatmap.png` | Correlation heatmap of student performance features (output of Task 2) |
| `final_score_distribution.png` | Distribution plot of final scores (output of Task 2) |
| `predicted_vs_actual.png` | Predicted vs. actual scatter plot (output of Task 3) |
| `confusion_matrix.png` | Confusion matrix for the sentiment classifier (output of Task 4) |

## How to Run

1. Open `vyrex_intern.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `student_performance.csv` and `sentiment_reviews.csv` to the working directory (`/content/` in Colab).
3. Run all cells in order — Task 2 will automatically generate `student_performance_clean.csv` and the plots, which Task 3 and Task 4 depend on.

## Limitations & Future Improvements

- No cross-validation was used — results are based on a single train/test split.
- No hyperparameter tuning was performed on either model.
- The sentiment dataset is relatively small; testing on a larger, more diverse real-world review dataset would give a more robust accuracy estimate.
- Future work could include polynomial/interaction features for the regression model to test for non-linear relationships, and a larger labeled dataset for the sentiment classifier.

## Acknowledgment

This project was completed as part of my **AI/ML Internship at Vyrex Technology**, as a practical demonstration of Python programming, data cleaning, exploratory data analysis, and supervised machine learning (regression and classification).
