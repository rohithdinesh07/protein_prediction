"""
ml_analysis.py
Rohith Dinesh
CSE 163

This module trains and evaluates a Decision Tree classifier
to predict protein functional group classifications using
engineered biochemical protein features.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data_processing import (
    load_dataset,
    calculate_biochemical_features
)


def ml_implementation(
    data: pd.DataFrame
) -> tuple[
    DecisionTreeClassifier,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """
    Trains a Decision Tree classifier to predict
    protein functional groups.

    Args:
        data: Protein DataFrame containing engineered
            biochemical features.

    Returns:
        A tuple containing:
            - trained DecisionTreeClassifier
            - feature DataFrame
            - X_test
            - y_test
            - predictions
    """
    labels = data["functional_group"]

    features = data[
        [
            "hydrophobic_ratio",
            "charged_pos_ratio",
            "charged_neg_ratio",
            "charge_balance",
            "polar_ratio",
            "aromatic_ratio",
            "Length",
            "Mass"
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.15,
        random_state=42
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, features, X_test, y_test, predictions


def graph_tree(
    model: DecisionTreeClassifier,
    features: pd.DataFrame
) -> None:
    """
    Creates and saves a visualization of the trained
    Decision Tree classifier.

    Args:
        model: Trained DecisionTreeClassifier.
        features: Feature DataFrame used for training.
    """
    plt.figure(figsize=(20, 10), dpi=300)

    plot_tree(
        model,
        feature_names=features.columns,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        impurity=False,
        proportion=True,
        rounded=True,
        max_depth=2,
        fontsize=6
    )

    plt.savefig(
        "decision_tree.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()


def evaluate_model(y_test: pd.Series, predictions: pd.Series) -> None:
    """
    Evaluates model performance using accuracy
    and classification metrics.

    Results are written to a text file.

    Args:
        y_test: Actual functional group labels.
        predictions: Predicted functional group labels.
    """
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    with open("ml_results.txt", "w", encoding="utf-8") as file:
        
        file.write(
            "Protein Functional Group "
            "Classification Results\n\n"
        )

        file.write(
            f"Model Accuracy: "
            f"{round(accuracy * 100, 2)}%\n\n"
        )

        file.write("Classification Report\n\n")

        file.write(report)


def main() -> None:
    """
    Loads the protein dataset, computes biochemical
    features, trains a Decision Tree classifier,
    evaluates model performance, and generates
    visualizations.
    """
    data = calculate_biochemical_features(load_dataset("protein_dataset.csv"))

    model, features, X_test, y_test, predictions = (ml_implementation(data))

    graph_tree(model, features)

    evaluate_model(y_test,predictions)


if __name__ == "__main__":
    main()
