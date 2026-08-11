# Student Performance & Sentiment Analysis — AI/ML Internship Project

**Internship Project — Vyrex Technology**

## Description

This repository contains the deliverables from my AI/ML internship at **Vyrex Technology**. The project is built across four progressive tasks that together demonstrate the core AI/ML development workflow — from fundamental programming logic to a complete supervised machine learning pipeline covering both regression and classification.

The project uses a student performance dataset and a product review dataset to answer two practical questions:
1. **Can we predict a student's final exam score** based on their study habits, attendance, sleep, and past performance?
2. **Can we automatically classify a piece of text (a review) as positive or negative sentiment**, without a human reading it?

Along the way, the project also demonstrates strong Python fundamentals and a full data cleaning / exploratory data analysis (EDA) pipeline — the essential building blocks of any real-world ML system.

## Objective

Machine learning models are only as good as the pipeline behind them. This project was designed to prove hands-on competency across the full pipeline an ML/AI role actually requires:
- Writing clean, correct, well-documented Python code
- Handling messy, real-world data (missing values, duplicates) responsibly
- Exploring data visually to form hypotheses before modeling
- Building, training, and rigorously evaluating a regression model
- Building, training, and rigorously evaluating a text classification model
- Interpreting model results honestly, including their limitations

## Project Structure

### Task 1: Python Fundamentals
Five standalone functions covering prime number generation, word frequency counting, list statistics, duplicate removal, and Fibonacci sequence generation — demonstrating core programming logic without relying on built-in shortcut libraries.

### Task 2: Data Cleaning & Exploratory Data Analysis (EDA)
Loads the raw student performance dataset (305 rows), inspects it for missing values and duplicates, cleans it (median imputation for missing numeric values, duplicate removal), and produces summary statistics and visualizations (correlation heatmap, score distribution). Outputs a clean dataset (300 rows) used for downstream modeling.

### Task 3: Linear Regression — Predicting Final Exam Scores

**Objective:** Predict a student's `final_score` (a continuous numeric value) using their study habits, attendance, sleep, and academic history — turning the cleaned dataset from Task 2 into an actual predictive model.

**Dataset used:** the cleaned output from Task 2 (`student_performance_clean.csv`), 300 rows, no missing values.

**Features (input variables, X):**
| Feature | Type | Description |
|---|---|---|
| `study_hours_per_day` | Numeric | Average hours studied per day |
| `attendance_percent` | Numeric | Class attendance percentage |
| `sleep_hours` | Numeric | Average hours slept per day |
| `previous_score` | Numeric | Score from a prior exam/assessment |
| `extracurricular` | Categorical (Yes/No) | Whether the student participates in extracurricular activities |

**Target (output variable, y):** `final_score` — the exam score the model is trying to predict.

**Preprocessing steps:**
1. **Categorical encoding** — `extracurricular` (Yes/No) was converted into a binary numeric column (`extracurricular_encoded`: 1 = Yes, 0 = No), since the model can only work with numbers.
2. **Train/test split** — the data was split 80% train / 20% test using `train_test_split` with `random_state=42` (fixed seed, so results are reproducible). The model only learns from the training set; the test set is held back to check how well it generalizes to data it has never seen.
3. **Feature scaling (StandardScaler)** — all features were standardized to have mean 0 and standard deviation 1. This doesn't change the model's predictions, but it makes the learned coefficients directly comparable in size, so you can fairly say "study hours matters more than sleep hours."

**Algorithm:** Ordinary Least Squares Linear Regression (`sklearn.linear_model.LinearRegression`). It models the target as a weighted sum of the input features:

```
final_score = (w1 × study_hours) + (w2 × attendance) + (w3 × sleep_hours) 
              + (w4 × previous_score) + (w5 × extracurricular) + intercept
```

The algorithm finds the weights (`w1`...`w5`) and intercept that minimize the sum of squared differences between predicted and actual scores on the training data.

**Evaluation metrics used (on the 20% held-out test set):**
| Metric | Value | What it means |
|---|---|---|
| R² (R-squared) | 0.651 | The model explains ~65% of the variation in final scores. 1.0 would be a perfect fit; 0 would mean the model is no better than always predicting the average score. |
| RMSE (Root Mean Squared Error) | 5.807 | Average prediction error, in score points, with larger errors penalized more heavily. |
| MAE (Mean Absolute Error) | 4.596 | Average absolute prediction error, in score points — a typical prediction is off by about 4.6 points. |

**Learned feature coefficients (standardized):**
| Feature | Coefficient | Interpretation |
|---|---|---|
| `study_hours_per_day` | +5.568 | Strongest positive driver of final score |
| `attendance_percent` | +3.822 | Second strongest positive driver |
| `previous_score` | +3.676 | Past performance is a solid predictor of future performance |
| `sleep_hours` | +1.818 | Positive but weaker effect |
| `extracurricular_encoded` | +0.257 | Minimal effect on final score |
| Intercept | 67.468 | Baseline predicted score when all standardized features are at their average value |

**Visualizations produced:**
- **Predicted vs. Actual scatter plot** (`predicted_vs_actual.png`) — plots the model's predictions against the true scores, with a red dashed "perfect prediction" line. Points close to this line indicate accurate predictions; the spread visible in this plot reflects the model's ~65% R² — good general trend-following, with more noticeable error at the extreme high and low ends of the score range.

**Interpretation / what this model tells us:** how much a student studies is, by a clear margin, the single biggest lever tied to their final score in this dataset — followed by how consistently they attend class and how they performed previously. Sleep and extracurricular involvement matter far less by comparison. This kind of insight (not just the prediction itself) is often the more valuable output of a regression model — it tells you *which* factors to focus on, not just *what* the score will be.

**Limitations specific to this model:**
- Linear regression assumes a straight-line relationship between each feature and the score — it can't capture effects like diminishing returns (e.g. studying 10 hours/day may not help proportionally more than studying 6 hours/day).
- Only one train/test split was evaluated — no k-fold cross-validation was performed, so the exact metric values could shift slightly with a different random seed.
- No hyperparameter tuning or regularization (e.g. Ridge/Lasso) was applied, since plain linear regression was sufficient for this dataset size and feature count.
- With an R² of 0.65, roughly 35% of the variation in final scores is driven by factors not captured in this dataset (e.g. teaching quality, exam difficulty, individual motivation).

### Task 4: Sentiment Classification — Positive vs Negative Reviews
Trains a Logistic Regression classifier on TF-IDF vectorized text (392 labeled reviews) to classify reviews as positive or negative. Evaluated using accuracy, precision, recall, F1-score, and a confusion matrix, then tested on brand-new unseen example sentences.

**Results:** Accuracy = 1.00 | Precision = 1.00 | Recall = 1.00 | F1 = 1.00
*Note: the near-perfect score reflects the size and relative simplicity of this dataset, rather than claiming a flawless, production-grade real-world sentiment classifier. Predictions on brand-new unseen sentences showed realistic, sub-1.0 confidence scores, which supports genuine generalization rather than memorization.*

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

## Limitations & Future Improvements

- No cross-validation was used — results are based on a single train/test split.
- No hyperparameter tuning was performed on either model.
- The sentiment dataset is relatively small; testing on a larger, more diverse real-world review dataset would give a more robust accuracy estimate.
- Future work could include polynomial/interaction features for the regression model to test for non-linear relationships, and a larger labeled dataset for the sentiment classifier.

## Acknowledgment

This project was completed as part of my **AI/ML Internship at Vyrex Technology**, as a practical demonstration of Python programming, data cleaning, exploratory data analysis, and supervised machine learning (regression and classification).
