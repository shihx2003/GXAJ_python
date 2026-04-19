import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_2d(data, title="2D Plot", xlabel="X", ylabel="Y", save_path=None):
    """
    Plot a 2D array/variable.
    
    Args:
        data: 2D numpy array or list
        title: Title of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
    """
    data = np.array(data)
    
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
def plot_line(data, title="Line Plot", xlabel="Date", ylabel="Y", save_path=None):
    """
    Plot multiple lines from a DataFrame.

    Args:
        data: pandas DataFrame, first column is Date, remaining columns are series
        title: Title of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if data.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 columns")

    x = data.iloc[:, 0]
    y_cols = data.columns[1:]

    plt.figure(figsize=(16, 6))

    for col in y_cols:
        plt.plot(x, data[col], label=str(col))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Example usage:
if __name__ == "__main__":
    data_2d = np.random.rand(10, 10)
    plot_2d(data_2d, title="Random 2D Data")