import numpy as np

import matplotlib.pyplot as plt

def plot_2d(data, title="2D Plot", xlabel="X", ylabel="Y"):
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
    plt.show()


# Example usage:
if __name__ == "__main__":
    data_2d = np.random.rand(10, 10)
    plot_2d(data_2d, title="Random 2D Data")