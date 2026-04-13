"""
Program Name: EECS 658 Assignment 6 - Part 1
Description:
    This program completes Part 1 of Assignment 6 using the full Iris dataset.
    It runs k-Means clustering for k = 1 through 20, plots reconstruction error
    vs. k, then evaluates clustering results for:
        1) k = elbow_k (chosen manually from the plot)
        2) k = 3

    The program prints confusion matrices and accuracy when appropriate.
    If elbow_k is not equal to 3, the program prints the confusion matrix but
    does not calculate accuracy, following the assignment instructions.

Inputs:
    None from the user during execution.
    The program uses the built-in Iris dataset from scikit-learn.

Outputs:
    - Plot of reconstruction error vs. k for k = 1 to 20
    - Confusion matrix for k = elbow_k
    - Accuracy for k = elbow_k if elbow_k = 3
    - Confusion matrix for k = 3
    - Accuracy for k = 3
    - Printed answer to Part 1 Question 1

Collaborators:
    None

Other Sources:
    Assignment 6 Instructions
    Rubric 6
    PlottingCode.py example
    ChatGPT

Author:
    Abhiroop Goel

Creation Date:
    2026-04-13
"""

# Import NumPy for numerical operations.
import numpy as np

# Import matplotlib for plotting.
import matplotlib.pyplot as plt

# Import the Iris dataset.
from sklearn.datasets import load_iris

# Import k-Means.
from sklearn.cluster import KMeans

# Import confusion_matrix for confusion matrix generation.
from sklearn.metrics import confusion_matrix

# Import Hungarian algorithm helper for best label matching.
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------
# IMPORTANT:
# Choose this value MANUALLY after looking at the reconstruction
# error plot. Run once, inspect the elbow, set the value here,
# then run again for final output.
# ------------------------------------------------------------------
ELBOW_K = 3


def print_label(title):
    """
    Print a clear label between outputs so the grader can easily
    see what is being displayed.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def plot_reconstruction_error(k_values, reconstruction_errors):
    """
    Plot reconstruction error vs. k using the style hinted in PlottingCode.py.
    """
    plt.figure()
    plt.plot(k_values, reconstruction_errors, marker='o')
    plt.title('Reconstruction Error vs. k')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.xticks(np.arange(1, 21, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_best_mapped_confusion_and_accuracy(y_true, y_pred):
    """
    Rearrange predicted cluster labels to maximize the diagonal of the
    confusion matrix. This lets us calculate the best possible mapping
    between arbitrary k-Means cluster labels and true class labels.

    Returns:
        raw_cm: confusion matrix using original cluster labels
        mapped_cm: confusion matrix after best mapping
        accuracy: diagonal sum / total after best mapping
    """
    # Build the raw 3x3 confusion matrix first.
    raw_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Convert confusion matrix to cost matrix for Hungarian algorithm.
    cost_matrix = raw_cm.max() - raw_cm

    # Find best one-to-one assignment.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a mapping from predicted cluster -> true class.
    cluster_to_class = {}
    for true_class_index, predicted_cluster_index in zip(row_ind, col_ind):
        cluster_to_class[predicted_cluster_index] = true_class_index

    # Remap predictions using the best assignment.
    remapped_predictions = np.array([cluster_to_class[label] for label in y_pred])

    # Build the mapped confusion matrix.
    mapped_cm = confusion_matrix(y_true, remapped_predictions, labels=[0, 1, 2])

    # Calculate accuracy.
    accuracy = np.trace(mapped_cm) / np.sum(mapped_cm)

    return raw_cm, mapped_cm, accuracy


def evaluate_kmeans_model(X, y, k, label_text):
    """
    Fit k-Means for a given k, predict cluster labels, and print the
    required confusion matrix and accuracy behavior.
    """
    # Create the k-Means model.
    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Fit the model on the full Iris dataset.
    model.fit(X)

    # Predict cluster assignments for the same dataset.
    predictions = model.predict(X)

    # Print a label before the output.
    print_label(label_text)

    # If k is 3, compute mapped confusion matrix and accuracy.
    if k == 3:
        raw_cm, mapped_cm, accuracy = build_best_mapped_confusion_and_accuracy(y, predictions)

        print("Raw confusion matrix using original cluster labels:")
        print(raw_cm)

        print("\nBest-mapped confusion matrix:")
        print(mapped_cm)

        print(f"\nAccuracy: {accuracy:.4f}")

    else:
        # If k is not 3, only print the built-in confusion matrix and the required message.
        # The assignment says accuracy should not be calculated in this case.
        cm = confusion_matrix(y, predictions, labels=list(range(k)))

        print("Confusion matrix:")
        print(cm)

        print("\nCannot calculate Accuracy Score because the number of classes")
        print("is not the same as the number of clusters.")


def main():
    """
    Main driver for Part 1.
    """
    # Load the Iris dataset.
    iris = load_iris()

    # Feature matrix.
    X = iris.data

    # True class labels.
    y = iris.target

    # Store k values from 1 through 20.
    k_values = list(range(1, 21))

    # Store reconstruction errors for each k.
    reconstruction_errors = []

    # Print label for the plot section.
    print_label("Part 1: Reconstruction Error vs. k")

    # Run k-Means for each k and store the inertia (reconstruction error).
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        reconstruction_errors.append(model.inertia_)

    # Print the numeric values too, which helps for the screenshot and grading.
    for k, err in zip(k_values, reconstruction_errors):
        print(f"k = {k:2d}, reconstruction error = {err:.4f}")

    # Plot reconstruction error vs. k.
    plot_reconstruction_error(k_values, reconstruction_errors)

    # Print manual elbow choice clearly.
    print_label("Part 1: Manual elbow selection")
    print(f"Manually selected elbow_k from the plot = {ELBOW_K}")

    # Evaluate at k = elbow_k.
    evaluate_kmeans_model(
        X,
        y,
        ELBOW_K,
        f"Part 1: Confusion Matrix and Accuracy for k = elbow_k = {ELBOW_K}"
    )

    # Evaluate at k = 3.
    evaluate_kmeans_model(
        X,
        y,
        3,
        "Part 1: Confusion Matrix and Accuracy for k = 3"
    )

    # Print answer to Question 1.
    print_label("Part 1: Question 1 Answer")

    if ELBOW_K == 3:
        print(
            "According to my results, the elbow appears at k = 3, so the results "
            "support the idea that there are 3 species of iris represented in the dataset."
        )
    else:
        print(
            f"According to my results, the elbow appears at k = {ELBOW_K}, not 3. "
            "So based on the k-Means reconstruction error curve, the data may suggest "
            "a clustering structure different from exactly 3 species."
        )


# Run the program.
if __name__ == "__main__":
    main()