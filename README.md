# Strategic Classification Project

## Codebase Structure

The project is organized as follows:

-   **`model.py`**: Defines the `StrategicClassifier` and other model variants (e.g., `StrategicClassifierForWarmup`, `StrategicClassifierFiniteSet`). These models integrate optimization layers (using `cvxpylayers`) to anticipate agent manipulation.
-   **`trainer.py`**: Contains the `StrategicTrainer` class, which manages the training loop, including loss calculation, optimization, and metric logging.
-   **`dataloader.py`**: Implements `SCPIDataset` and data loading utilities (`create_dataloaders`) for handling datasets efficiently with PyTorch.
-   **`model_utils.py`**: Provides utility functions and custom loss modules (e.g., `BasicStrategicHingeLoss`, `AmbiguousStrategicHingeLoss`) and helper functions for calculating utility and burden.
-   **`data_utils.py`**: Utilities for data processing.
-   **`synthetic_exp.ipynb`**: A generic Jupyter Notebook demonstrating how to generate synthetic data, train the model, and visualize results.
-   **`data/`**: Directory containing datasets, organized by source (e.g., `real_data`, `synthetic_data`).

## Installation Guide

### Prerequisites

-   [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
-   Git (optional, for cloning the repository).

### Setting up the Environment

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create the Conda environment**:
    The project uses an `environment.yml` file to manage dependencies. Run the following command to create the environment named `env`:
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment**:
    ```bash
    conda activate env
    ```

### Dependencies

Key dependencies included in the environment:
-   Python 3.11
-   PyTorch = 2.6.0
-   CVXPY >= 1.3.0 & CVXPY Layers
-   Scikit-learn, Numpy, Pandas
-   Matplotlib