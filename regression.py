"""
Task 3: Linear Regression Model
----------------------------------
Trains a Linear Regression model to predict a student's final exam
score from study habits and attendance, then reports RMSE (and a few
other useful metrics) on a held-out test set.

Dataset: ../data/student_performance_clean.csv (produced by Task 2's eda.py)
Falls back to the raw CSV with basic cleaning if the clean file isn't found,
so this script can also be run standalone.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLEAN_PATH = DATA_DIR / "student_performance_clean.csv"
RAW_PATH = DATA_DIR / "student_performance.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset if available, otherwise clean the raw one on the fly."""
    if CLEAN_PATH.exists():
        return pd.read_csv(CLEAN_PATH)

    df = pd.read_csv(RAW_PATH).drop_duplicates()
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame):
    """Select feature columns (X) and target (y). Encodes the categorical column."""
    df = df.copy()
    df["extracurricular_encoded"] = (df["extracurricular"] == "Yes").astype(int)

    feature_cols = [
        "study_hours_per_day",
        "attendance_percent",
        "sleep_hours",
        "previous_score",
        "extracurricular_encoded",
    ]
    X = df[feature_cols]
    y = df["final_score"]
    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    """Split data, train a Linear Regression model, and evaluate on the test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling isn't strictly required for linear regression, but it makes the
    # learned coefficients directly comparable to each other.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=" * 60)
    print("MODEL EVALUATION (on held-out test set, 20% of data)")
    print("=" * 60)
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"MAE  (Mean Absolute Error):     {mae:.3f}")
    print(f"R^2  (Coefficient of Determination): {r2:.3f}")

    print("\nLearned coefficients (standardized features):")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name:28s}: {coef:+.3f}")
    print(f"  {'intercept':28s}: {model.intercept_:+.3f}")

    return model, scaler, X_test, y_test, y_pred, rmse


def plot_predictions(y_test, y_pred):
    """Save a predicted-vs-actual scatter plot."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title("Linear Regression: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR / 'predicted_vs_actual.png'}")


def main():
    df = load_dataset()
    X, y, feature_cols = prepare_features(df)
    model, scaler, X_test, y_test, y_pred, rmse = train_and_evaluate(X, y, feature_cols)
    plot_predictions(y_test, y_pred)


if __name__ == "__main__":
    main()
