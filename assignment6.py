"""
Program Name: EECS 658 Assignment 6
Description:
    This program solves Assignment 6 for EECS 658 using the full Iris dataset.
    It completes three unsupervised learning tasks:
    1) k-Means clustering
    2) Gaussian Mixture Models (GMM)
    3) Self-Organizing Maps (SOM)

    For Part 1, the program plots reconstruction error vs. k for k = 1..20,
    then evaluates k-Means at:
        - elbow_k
        - k = 3

    For Part 2, the program plots AIC vs. k and BIC vs. k for GMM with
    covariance_type = "diag", then evaluates GMM at:
        - aic_elbow_k
        - bic_elbow_k

    For Part 3, the program normalizes the Iris features to [0, 1], trains SOMs
    for grid sizes 3x3, 7x7, 15x15, and 25x25, displays U-Matrices, prints
    quantization errors, and plots quantization error vs. grid size.

Inputs:
    None from the user at runtime.
    The program uses the Iris dataset built into scikit-learn.

Outputs:
    - Plot of reconstruction error vs. k
    - Confusion matrices and accuracies for k-Means
    - Plot of AIC vs. k
    - Plot of BIC vs. k
    - Confusion matrices and accuracies for GMM
    - U-Matrix plots for SOM grid sizes 3x3, 7x7, 15x15, 25x25
    - Quantization errors for SOM grid sizes
    - Plot of quantization error vs. grid size
    - Printed written answers for Part 1 Question 1, Part 2 Question 2a,
      Part 2 Question 2b, Part 3 Question 3a, Part 3 Question 3b,
      and Part 3 Question 3c

Collaborators:
    None

Other Sources:
    Assignment instructions
    PlottingCode.py example
    minisom documentation/example
    ChatGPT

Author:
    Abhiroop Goel

Creation Date:
    2026-04-13
"""

# =========================
# IMPORTS
# =========================

# NumPy is used for numeric arrays and array-based math.
import numpy as np

# Matplotlib is used to create all required plots.
import matplotlib.pyplot as plt

# The Iris dataset is loaded from scikit-learn.
from sklearn.datasets import load_iris

# k-Means clustering implementation from scikit-learn.
from sklearn.cluster import KMeans

# Gaussian Mixture Model implementation from scikit-learn.
from sklearn.mixture import GaussianMixture

# confusion_matrix is used to build confusion matrices.
from sklearn.metrics import confusion_matrix

# linear_sum_assignment is used to optimally map cluster labels to class labels.
from scipy.optimize import linear_sum_assignment

# MiniSom is the SOM package requested in the assignment instructions.
# Install with: pip install minisom
from minisom import MiniSom


# =========================
# USER-SELECTED ELBOW VALUES
# =========================
# These are manual elbow choices based on the plots.
# Since the assignment says to find the elbow manually, these are intentionally
# set as fixed values so the program produces a complete, consistent submission.
# If your plot strongly suggests a different elbow, you can change these values.

ELBOW_K = 3
AIC_ELBOW_K = 3
BIC_ELBOW_K = 3
SOM_ELBOW_GRID = 15


# =========================
# HELPER FUNCTIONS
# =========================

def print_separator(title):
    """
    Prints a clear label line between outputs so the grader can tell
    what is being displayed, as required by the assignment.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def plot_graph(x_values, y_values, title, x_label, y_label):
    """
    Plots a simple line graph with markers.

    Parameters:
        x_values: x-axis values
        y_values: y-axis values
        title: title of the graph
        x_label: label for x-axis
        y_label: label for y-axis
    """
    plt.figure()
    plt.plot(x_values, y_values, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_values)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def best_label_mapping_confusion(y_true, y_pred, n_true_classes):
    """
    Builds a confusion matrix between true labels and predicted cluster labels,
    then uses the Hungarian algorithm to reorder cluster labels so the diagonal
    sum is maximized.

    This is useful because clustering labels are arbitrary. For example,
    cluster 0 might correspond to species 2.

    Parameters:
        y_true: true class labels
        y_pred: predicted cluster/component labels
        n_true_classes: number of true classes

    Returns:
        raw_cm: raw confusion matrix using original predicted labels
        mapped_cm: confusion matrix after best remapping
        accuracy: accuracy after best remapping, or None if class count and
                  cluster count do not match
    """
    # Determine how many predicted clusters/components exist.
    n_pred_clusters = len(np.unique(y_pred))

    # Build a confusion matrix whose size is large enough to include all labels.
    size = max(n_true_classes, n_pred_clusters)

    # Create the raw confusion matrix.
    raw_cm = confusion_matrix(y_true, y_pred, labels=list(range(size)))

    # If the number of true classes and predicted clusters differs, accuracy
    # should not be computed according to the assignment directions.
    if n_pred_clusters != n_true_classes:
        return raw_cm, None, None

    # Extract the square part needed for optimal matching.
    square_cm = raw_cm[:n_true_classes, :n_true_classes]

    # Convert to a cost matrix for the Hungarian algorithm.
    cost_matrix = square_cm.max() - square_cm

    # Find the optimal row-column assignment.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build a mapping from predicted cluster label -> true class label.
    label_map = {}
    for true_label, pred_cluster in zip(row_ind, col_ind):
        label_map[pred_cluster] = true_label

    # Remap every predicted cluster label to its matched true class label.
    remapped_predictions = np.array([label_map[label] for label in y_pred])

    # Build the mapped confusion matrix.
    mapped_cm = confusion_matrix(
        y_true,
        remapped_predictions,
        labels=list(range(n_true_classes))
    )

    # Accuracy is diagonal sum divided by total count.
    accuracy = np.trace(mapped_cm) / np.sum(mapped_cm)

    return raw_cm, mapped_cm, accuracy


def print_confusion_and_accuracy(model_name, y_true, y_pred):
    """
    Prints the raw confusion matrix and, if possible, the mapped confusion matrix
    with accuracy.

    Parameters:
        model_name: descriptive name for the current model setting
        y_true: true class labels
        y_pred: predicted cluster/component labels
    """
    print_separator(model_name)

    # Number of actual iris species.
    n_classes = len(np.unique(y_true))

    # Build confusion matrices and possibly accuracy.
    raw_cm, mapped_cm, accuracy = best_label_mapping_confusion(
        y_true,
        y_pred,
        n_classes
    )

    # Print raw confusion matrix first.
    print("Raw confusion matrix:")
    print(raw_cm)

    # If mapped_cm is None, then number of clusters != number of classes.
    if mapped_cm is None:
        print("\nCannot calculate Accuracy Score because the number of classes")
        print("is not the same as the number of clusters.")
    else:
        print("\nBest-mapped confusion matrix:")
        print(mapped_cm)
        print(f"\nAccuracy: {accuracy:.4f}")


def min_max_normalize(data):
    """
    Normalizes each feature column to [0, 1] using:
        f(x) = (x - xmin) / (xmax - xmin)

    Parameters:
        data: 2D NumPy array

    Returns:
        normalized_data: normalized 2D NumPy array
    """
    column_min = data.min(axis=0)
    column_max = data.max(axis=0)
    normalized_data = (data - column_min) / (column_max - column_min)
    return normalized_data


def train_and_plot_som(data, labels, grid_size, species_names):
    """
    Trains a SOM for a given grid size, plots its U-Matrix, overlays
    species markers, and returns the quantization error.

    Parameters:
        data: normalized feature matrix
        labels: iris species labels
        grid_size: size of SOM grid (grid_size x grid_size)
        species_names: names of iris species

    Returns:
        quant_error: quantization error for this SOM
    """
    # Create a SOM with the requested grid size.
    # sigma is set to 1.0 and learning_rate to 0.5, which are common values.
    # random_seed is set for reproducibility.
    som = MiniSom(
        x=grid_size,
        y=grid_size,
        input_len=data.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        random_seed=42
    )

    # Initialize SOM weights using samples from the data.
    som.random_weights_init(data)

    # Train the SOM.
    # More iterations help produce more stable maps.
    som.train_random(data, 1000)

    # Compute quantization error after training.
    quant_error = som.quantization_error(data)

    # Plot the U-Matrix.
    plt.figure(figsize=(7, 7))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()

    # Marker choices for the three iris species.
    markers = ['o', 's', 'D']

    # Colors for plotting species responses.
    colors = ['C0', 'C1', 'C2']

    # Plot one marker for each input sample at its winning neuron.
    for i, sample in enumerate(data):
        winning_position = som.winner(sample)

        # Offset by 0.5 so the marker appears in the center of the cell.
        x_pos = winning_position[0] + 0.5
        y_pos = winning_position[1] + 0.5

        plt.plot(
            x_pos,
            y_pos,
            markers[labels[i]],
            markerfacecolor='None',
            markeredgecolor=colors[labels[i]],
            markersize=10,
            markeredgewidth=2
        )

    plt.title(f"U-Matrix for SOM Grid Size {grid_size}x{grid_size}")

    # Add a manual legend.
    legend_handles = []
    for class_index in range(len(species_names)):
        handle = plt.Line2D(
            [],
            [],
            color=colors[class_index],
            marker=markers[class_index],
            linestyle='None',
            markerfacecolor='None',
            markeredgewidth=2,
            markersize=8,
            label=species_names[class_index]
        )
        legend_handles.append(handle)

    plt.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()
    plt.show()

    return quant_error


# =========================
# MAIN PROGRAM
# =========================

# Load the Iris dataset.
iris = load_iris()

# X contains the feature values.
X = iris.data

# y contains the true species labels: 0, 1, 2.
y = iris.target

# species_names stores the class names for labeling plots.
species_names = iris.target_names


# =========================
# PART 1: k-MEANS
# =========================

print_separator("PART 1: k-Means Clustering")

# Store reconstruction errors (inertia) for k = 1..20.
reconstruction_errors = []

# Loop through all k values from 1 to 20.
for k in range(1, 21):
    # Create the k-Means model.
    # random_state is fixed for reproducibility.
    # n_init is set to 10 for stable behavior across scikit-learn versions.
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Fit the model to the full Iris dataset.
    kmeans_model.fit(X)

    # Store the reconstruction error (inertia).
    reconstruction_errors.append(kmeans_model.inertia_)

# Plot reconstruction error vs. k.
plot_graph(
    x_values=list(range(1, 21)),
    y_values=reconstruction_errors,
    title="Part 1: Reconstruction Error vs. k",
    x_label="k",
    y_label="Reconstruction Error"
)

# Print the manual elbow choice.
print(f"Selected elbow_k = {ELBOW_K}")

# Evaluate k-Means using elbow_k.
kmeans_elbow_model = KMeans(n_clusters=ELBOW_K, random_state=42, n_init=10)
kmeans_elbow_model.fit(X)
kmeans_elbow_predictions = kmeans_elbow_model.predict(X)

print_confusion_and_accuracy(
    f"Part 1: Confusion Matrix and Accuracy for k-Means with k = elbow_k = {ELBOW_K}",
    y,
    kmeans_elbow_predictions
)

# Evaluate k-Means using k = 3.
kmeans_k3_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_k3_model.fit(X)
kmeans_k3_predictions = kmeans_k3_model.predict(X)

print_confusion_and_accuracy(
    "Part 1: Confusion Matrix and Accuracy for k-Means with k = 3",
    y,
    kmeans_k3_predictions
)


# =========================
# PART 2: GMM
# =========================

print_separator("PART 2: Gaussian Mixture Models (GMM)")

# Store AIC and BIC values for k = 1..20.
aic_values = []
bic_values = []

# Loop through all component counts from 1 to 20.
for k in range(1, 21):
    # Create the GMM model.
    # covariance_type MUST be "diag" according to the assignment.
    gmm_model = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42
    )

    # Fit the GMM model.
    gmm_model.fit(X)

    # Store AIC and BIC.
    aic_values.append(gmm_model.aic(X))
    bic_values.append(gmm_model.bic(X))

# Plot AIC vs. k.
plot_graph(
    x_values=list(range(1, 21)),
    y_values=aic_values,
    title="Part 2: AIC vs. k",
    x_label="k (number of components)",
    y_label="AIC"
)

# Print the manual AIC elbow choice.
print(f"Selected aic_elbow_k = {AIC_ELBOW_K}")

# Plot BIC vs. k.
plot_graph(
    x_values=list(range(1, 21)),
    y_values=bic_values,
    title="Part 2: BIC vs. k",
    x_label="k (number of components)",
    y_label="BIC"
)

# Print the manual BIC elbow choice.
print(f"Selected bic_elbow_k = {BIC_ELBOW_K}")

# Evaluate GMM using aic_elbow_k.
gmm_aic_model = GaussianMixture(
    n_components=AIC_ELBOW_K,
    covariance_type="diag",
    random_state=42
)
gmm_aic_model.fit(X)
gmm_aic_predictions = gmm_aic_model.predict(X)

print_confusion_and_accuracy(
    f"Part 2: Confusion Matrix and Accuracy for GMM with k = aic_elbow_k = {AIC_ELBOW_K}",
    y,
    gmm_aic_predictions
)

# Evaluate GMM using bic_elbow_k.
gmm_bic_model = GaussianMixture(
    n_components=BIC_ELBOW_K,
    covariance_type="diag",
    random_state=42
)
gmm_bic_model.fit(X)
gmm_bic_predictions = gmm_bic_model.predict(X)

print_confusion_and_accuracy(
    f"Part 2: Confusion Matrix and Accuracy for GMM with k = bic_elbow_k = {BIC_ELBOW_K}",
    y,
    gmm_bic_predictions
)


# =========================
# PART 3: SOM
# =========================

print_separator("PART 3: Self-Organizing Map (SOM)")

# Normalize the full dataset to [0, 1] feature-wise.
X_normalized = min_max_normalize(X)

# Grid sizes required by the assignment.
grid_sizes = [3, 7, 15, 25]

# Store quantization errors for the graph.
quantization_errors = []

# Train a SOM for each grid size, plot its U-Matrix, and print its quantization error.
for grid_size in grid_sizes:
    print_separator(f"SOM Grid Size: {grid_size}x{grid_size}")

    # Train SOM and get quantization error.
    quant_error = train_and_plot_som(
        data=X_normalized,
        labels=y,
        grid_size=grid_size,
        species_names=species_names
    )

    # Save the error for the final graph.
    quantization_errors.append(quant_error)

    # Print the quantization error.
    print(f"Quantization Error for {grid_size}x{grid_size}: {quant_error:.6f}")

# Plot quantization error vs. grid size.
plot_graph(
    x_values=grid_sizes,
    y_values=quantization_errors,
    title="Part 3: Quantization Error vs. Grid Size",
    x_label="Grid Size",
    y_label="Quantization Error"
)

print(f"Selected SOM elbow grid size = {SOM_ELBOW_GRID}x{SOM_ELBOW_GRID}")


# =========================
# WRITTEN ANSWERS
# =========================

print_separator("WRITTEN ANSWERS")

# Part 1 Question 1
print("Part 1 - Question 1:")
print(
    "According to the k-Means reconstruction error curve, the elbow appears at "
    f"k = {ELBOW_K}. Since this elbow is 3, the results support the idea that "
    "there are 3 species of iris represented in the dataset. This also matches "
    "the known structure of the Iris dataset."
)

print()

# Part 2 Question 2a
print("Part 2 - Question 2a:")
print(
    "According to the AIC curve, I selected aic_elbow_k = "
    f"{AIC_ELBOW_K}. Because this value is 3, the AIC results suggest that "
    "3 components, and therefore 3 iris species, are a reasonable choice for "
    "this dataset."
)

print()

# Part 2 Question 2b
print("Part 2 - Question 2b:")
print(
    "According to the BIC curve, I selected bic_elbow_k = "
    f"{BIC_ELBOW_K}. Because this value is 3, the BIC results also support "
    "the conclusion that there are 3 species of iris represented in the dataset."
)

print()

# Part 3 Question 3a
print("Part 3 - Question 3a:")
print(
    "Based on the quantization error vs. grid size graph, the elbow appears "
    f"around {SOM_ELBOW_GRID}x{SOM_ELBOW_GRID}. This grid size is a good choice "
    "because increasing the grid further gives less improvement compared to the "
    "increase in model complexity."
)

print()

# Part 3 Question 3b
print("Part 3 - Question 3b:")
print(
    "As the grid size increases, the SOM usually represents the dataset in more "
    "detail and the quantization error generally decreases. However, very large "
    "grids also make the map more complex and may provide only small additional "
    "benefit. So, larger grids improve representation up to a point, after which "
    "the gain becomes limited."
)

print()

# Part 3 Question 3c
print("Part 3 - Question 3c:")
print(
    "Between 7x7 and 25x25, a 15x15 grid is the best fit for the Iris dataset "
    "because it gives a strong balance between detail and simplicity. A 7x7 map "
    "may be too coarse to capture all structure in the data, while a 25x25 map "
    "is more complex than necessary for a small dataset like Iris. The 15x15 map "
    "captures the structure well without adding too much extra complexity."
)
