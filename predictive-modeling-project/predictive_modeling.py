"""Reproducible supervised-learning examples for classification and regression."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SEED = 42
TEST_SIZE = 0.20
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def run_classification() -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Compare classifiers on the built-in Wisconsin breast cancer dataset."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=dataset.target,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }
    rows = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        # Dataset convention: 0 = malignant, 1 = benign. Use benign as the
        # positive class consistently for precision, recall, and ROC/AUC.
        probability = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "precision_benign": precision_score(y_test, predicted, pos_label=1),
                "recall_benign": recall_score(y_test, predicted, pos_label=1),
                "roc_auc_benign": auc(*roc_curve(y_test, probability)[:2]),
            }
        )
        predictions[name] = (predicted, probability)

    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Breast Cancer Classification — Held-out Test Set", fontsize=16, weight="bold")
    for ax, (name, (predicted, probability)) in zip(axes.flat[:3], predictions.items()):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, predicted), display_labels=dataset.target_names
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=15)
    ax = axes.flat[3]
    for name, (_, probability) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        model_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={model_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set(title="ROC Curves (positive class: benign)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ARTIFACTS / "classification_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return results, predictions


def run_regression() -> pd.DataFrame:
    """Compare regressors on the built-in diabetes progression dataset."""
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=TEST_SIZE, random_state=SEED
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
    }
    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append(
            {
                "model": name,
                "r2": r2_score(y_test, predicted),
                "rmse": mean_squared_error(y_test, predicted) ** 0.5,
            }
        )
    results = pd.DataFrame(rows).sort_values("r2", ascending=False)

    best_name = results.iloc[0]["model"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Diabetes Progression Regression — Held-out Test Set", fontsize=15, weight="bold")
    axes[0].scatter(y_test, predictions[best_name], alpha=0.75, edgecolors="none")
    low, high = float(np.min(y_test)), float(np.max(y_test))
    axes[0].plot([low, high], [low, high], linestyle="--", color="firebrick")
    axes[0].set(title=f"Actual vs. predicted ({best_name})", xlabel="Actual target", ylabel="Predicted target")
    axes[1].bar(results["model"], results["r2"], color=["#4472C4", "#70AD47", "#ED7D31"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="R² by model", ylabel="R² (higher is better)")
    axes[1].tick_params(axis="x", labelrotation=18)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ARTIFACTS / "regression_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return results


def main() -> None:
    print("Welcome to your machine-learning practice project!\n")
    print("First, we will predict a label (malignant or benign). Then, we will predict a number.")
    print("Each model learns from 80% of the examples and is scored on the other 20%.\n")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    classification, _ = run_classification()
    regression = run_regression()
    classification.to_csv(ARTIFACTS / "classification_metrics.csv", index=False)
    regression.to_csv(ARTIFACTS / "regression_metrics.csv", index=False)
    summary = {
        "random_seed": SEED,
        "test_fraction": TEST_SIZE,
        "classification": classification.to_dict(orient="records"),
        "regression": regression.to_dict(orient="records"),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CLASSIFICATION: choosing a label")
    print("Practice data: breast cell measurements. Labels: malignant (0) and benign (1).")
    print(classification.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nREGRESSION: estimating a number")
    print("Practice data: measurements paired with a diabetes progression score.")
    print(regression.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nYour charts and score tables are ready in the 'artifacts' folder.")
    print(f"Folder location: {ARTIFACTS}")
    print("Tip: open README.md for help understanding accuracy, confusion matrices, ROC AUC, R², and RMSE.")


if __name__ == "__main__":
    main()
