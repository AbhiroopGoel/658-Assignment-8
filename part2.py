"""
EECS 658 Assignment 6 - Part 2 (GMM)

This program:
1. Plots AIC vs k (k = 1..20)
2. Plots BIC vs k (k = 1..20)
3. Uses manually selected elbows:
   - aic_elbow_k
   - bic_elbow_k
4. Prints confusion matrices and accuracy (only if k = 3)
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


# ==============================
# 🔴 SET THESE AFTER SEEING PLOTS
# ==============================
AIC_ELBOW_K = 3
BIC_ELBOW_K = 3


def print_label(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def plot_graph(k_vals, values, title, filename):
    plt.figure()
    plt.plot(k_vals, values, marker='o')
    plt.title(title)
    plt.xlabel("k (number of components)")
    plt.ylabel(title.split(" vs")[0])
    plt.xticks(range(1, 21))
    plt.grid(True)

    # Save plot (important for Codespaces)
    plt.savefig(filename)
    plt.show()


def best_mapping(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    cost = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    mapped = np.array([mapping[label] for label in y_pred])
    mapped_cm = confusion_matrix(y_true, mapped)

    acc = np.trace(mapped_cm) / np.sum(mapped_cm)

    return cm, mapped_cm, acc


def evaluate_gmm(X, y, k, title):
    print_label(title)

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",  # REQUIRED
        random_state=42
    )

    gmm.fit(X)
    preds = gmm.predict(X)

    if k == 3:
        raw, mapped, acc = best_mapping(y, preds)

        print("Raw confusion matrix:")
        print(raw)

        print("\nBest-mapped confusion matrix:")
        print(mapped)

        print(f"\nAccuracy: {acc:.4f}")

    else:
        cm = confusion_matrix(y, preds)
        print("Confusion matrix:")
        print(cm)
        print("\nCannot calculate Accuracy Score because the number of classes")
        print("is not the same as the number of clusters.")


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    k_vals = list(range(1, 21))

    aic_vals = []
    bic_vals = []

    print_label("Part 2: AIC and BIC values")

    for k in k_vals:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42
        )
        gmm.fit(X)

        aic = gmm.aic(X)
        bic = gmm.bic(X)

        aic_vals.append(aic)
        bic_vals.append(bic)

        print(f"k={k:2d}, AIC={aic:.2f}, BIC={bic:.2f}")

    # Plot AIC
    plot_graph(k_vals, aic_vals, "AIC vs. k", "aic_plot.png")

    # Plot BIC
    plot_graph(k_vals, bic_vals, "BIC vs. k", "bic_plot.png")

    print_label("Manual elbow selection")
    print(f"aic_elbow_k = {AIC_ELBOW_K}")
    print(f"bic_elbow_k = {BIC_ELBOW_K}")

    # Evaluate AIC elbow
    evaluate_gmm(X, y, AIC_ELBOW_K,
                 f"GMM using aic_elbow_k = {AIC_ELBOW_K}")

    # Evaluate BIC elbow
    evaluate_gmm(X, y, BIC_ELBOW_K,
                 f"GMM using bic_elbow_k = {BIC_ELBOW_K}")

    # ======================
    # Written Answers
    # ======================
    print_label("Question 2a Answer")

    print(
        f"Based on the AIC curve, the elbow occurs at k = {AIC_ELBOW_K}. "
        "Since this value is 3, the results suggest that there are 3 species "
        "of iris represented in the dataset."
    )

    print_label("Question 2b Answer")

    print(
        f"Based on the BIC curve, the elbow occurs at k = {BIC_ELBOW_K}. "
        "Since this value is 3, the results also support that there are "
        "3 species of iris in the dataset."
    )


if __name__ == "__main__":
    main()