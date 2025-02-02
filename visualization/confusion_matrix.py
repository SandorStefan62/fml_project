import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, class_names=None, output_dir="plots", name="model"):
    """
    Generate and save a confusion matrix plot.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_names: List of class names for axis labels
    - output_dir: Directory to save the plot image
    - name: Used to name the plot file
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in np.unique(y_true)]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f"{output_dir}/confusion_matrix_{name}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Confusion matrix plot saved at {plot_path}")