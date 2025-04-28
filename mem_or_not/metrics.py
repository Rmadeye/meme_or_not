from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    fig.suptitle(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    return fig


def calculate_confusion_matrix(
    y_true: list,
    y_pred: list,
) -> tuple:
    """
    Calculate the confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        labels (list): List of class labels.

    Returns:
        tuple: Confusion matrix and class labels.
    """

    cm = confusion_matrix(y_true, y_pred)
    return cm


def save_evaluation_metrics(results, train_cfg):
    """Save evaluation metrics to files."""
    cm = confusion_matrix(results["y_true"], results["y_pred"], labels=[0, 1])
    cm_fig = plot_confusion_matrix(
        cm,
        labels=["Not Meme", "Meme"],
        title="Confusion Matrix",
    )
    cm_fig.savefig(Path(train_cfg["model_dir"]) / "confusion_matrix.png")
    cr = classification_report(
        results["y_true"],
        results["y_pred"],
        target_names=["Not Meme", "Meme"],
        output_dict=True,
    )
    cr_df = pd.DataFrame(cr).T
    cr_df.to_csv(Path(train_cfg["model_dir"]) / "classification_report.csv")
    return cm_fig, cr
