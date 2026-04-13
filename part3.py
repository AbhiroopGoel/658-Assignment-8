"""
EECS 658 Assignment 6 - Part 3 (SOM)

This program:
1. Loads the Iris dataset
2. Normalizes each feature to [0, 1] using min-max normalization
3. Trains MiniSom SOMs for grid sizes:
   3x3, 7x7, 15x15, 25x25
4. Plots the U-Matrix for each grid size with species markers
5. Prints quantization error for each grid size
6. Plots quantization error vs. grid size
7. Prints written answers for Questions 3a, 3b, and 3c
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from minisom import MiniSom


# Manual elbow choice after looking at quantization error graph
SELECTED_SOM_GRID = 15


def print_label(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def min_max_normalize(data):
    """
    Normalize each feature column to [0, 1] using:
    f(x) = (x - xmin) / (xmax - xmin)
    """
    xmin = np.min(data, axis=0)
    xmax = np.max(data, axis=0)
    normalized = (data - xmin) / (xmax - xmin)
    return normalized


def plot_umatrix_with_markers(som, data, labels, species_names, grid_size, filename):
    """
    Plot the SOM U-Matrix and overlay species markers.
    """
    plt.figure(figsize=(8, 8))

    # Distance map (U-Matrix)
    umatrix = som.distance_map()
    plt.pcolor(umatrix.T, cmap='bone_r')
    plt.colorbar()

    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']

    for i, sample in enumerate(data):
        winner = som.winner(sample)

        # Center marker in the cell
        x = winner[0] + 0.5
        y = winner[1] + 0.5

        plt.plot(
            x,
            y,
            markers[labels[i]],
            markerfacecolor='None',
            markeredgecolor=colors[labels[i]],
            markersize=10,
            markeredgewidth=2
        )

    legend_handles = []
    for idx, name in enumerate(species_names):
        handle = plt.Line2D(
            [],
            [],
            color=colors[idx],
            marker=markers[idx],
            linestyle='None',
            markerfacecolor='None',
            markeredgewidth=2,
            markersize=8,
            label=name
        )
        legend_handles.append(handle)

    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(f"U-Matrix for SOM Grid Size {grid_size}x{grid_size}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_quantization_error(grid_sizes, quant_errors, filename):
    """
    Plot quantization error vs. grid size.
    """
    plt.figure()
    plt.plot(grid_sizes, quant_errors, marker='o')
    plt.title("Quantization Error vs. Grid Size")
    plt.xlabel("Grid Size")
    plt.ylabel("Quantization Error")
    plt.xticks(grid_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def train_som_and_get_error(data, labels, species_names, grid_size):
    """
    Train one SOM, save its U-Matrix plot, and return quantization error.
    """
    som = MiniSom(
        x=grid_size,
        y=grid_size,
        input_len=data.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        random_seed=42
    )

    som.random_weights_init(data)

    # Enough iterations for a small dataset like Iris
    som.train_random(data, 1000)

    quant_error = som.quantization_error(data)

    filename = f"umatrix_{grid_size}x{grid_size}.png"
    plot_umatrix_with_markers(som, data, labels, species_names, grid_size, filename)

    return quant_error


def main():
    print_label("Part 3: Self-Organizing Map (SOM)")

    iris = load_iris()
    X = iris.data
    y = iris.target
    species_names = iris.target_names

    # Normalize using the required min-max normalization formula
    X_norm = min_max_normalize(X)

    print_label("Normalized Data Check")
    print("First 5 rows of normalized Iris data:")
    print(X_norm[:5])

    grid_sizes = [3, 7, 15, 25]
    quant_errors = []

    for grid_size in grid_sizes:
        print_label(f"SOM Grid Size: {grid_size}x{grid_size}")

        quant_error = train_som_and_get_error(
            X_norm,
            y,
            species_names,
            grid_size
        )

        quant_errors.append(quant_error)
        print(f"Quantization Error for {grid_size}x{grid_size}: {quant_error:.6f}")

    print_label("Quantization Errors Summary")
    for g, q in zip(grid_sizes, quant_errors):
        print(f"{g}x{g}: {q:.6f}")

    plot_quantization_error(
        grid_sizes,
        quant_errors,
        "quantization_error_vs_grid_size.png"
    )

    print_label("Manual Elbow Selection for SOM")
    print(f"Selected SOM elbow grid size = {SELECTED_SOM_GRID}x{SELECTED_SOM_GRID}")

    print_label("Question 3a Answer")
    print(
        f"Based on the quantization error vs. grid size plot, I would select "
        f"{SELECTED_SOM_GRID}x{SELECTED_SOM_GRID} as the elbow grid size because "
        "it provides a strong reduction in quantization error before the improvements "
        "begin to level off."
    )

    print_label("Question 3b Answer")
    print(
        "As the grid size increases, the SOM is able to represent the structure of "
        "the data more precisely, so the quantization error generally decreases. "
        "However, larger grids also make the map more complex, and after a certain "
        "point the improvement becomes smaller. This means increasing the grid size "
        "helps performance up to a point, but very large grids may add complexity "
        "without much extra benefit."
    )

    print_label("Question 3c Answer")
    print(
        "Between 7x7 and 25x25, a 15x15 grid is the best fit for the Iris dataset "
        "because it balances detail and simplicity. A 7x7 grid may be too small to "
        "capture all of the structure in the data, while a 25x25 grid is likely more "
        "complex than necessary for a small dataset like Iris. A 15x15 grid gives a "
        "good representation without adding too much unnecessary complexity."
    )


if __name__ == "__main__":
    main()