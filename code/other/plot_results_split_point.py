import matplotlib.pyplot as plt # type: ignore
import json
import numpy as np # type: ignore
import time
import seaborn as sns # type: ignore

file_title = 'diff-config-long'
# Load data from JSON files
with open('../data/' + file_title + '-results-mul-runs-split.json', 'r') as file:
    data = json.load(file)

with open('../data/' + file_title + '-merged-results-mul-runs-split.json', 'r') as file:
    merged_data = json.load(file)

with open('../data/' + file_title + '-naive-results-mul-runs-split.json', 'r') as file:
    naive_data = json.load(file)

# Nodes to plot
#nodes_to_plot = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
nodes_to_plot = ["m3-123", "m3-133", "m3-143", "m3-150", "m3-159", "m3-163", "m3-166"]
#nodes_to_plot = ["m3-123"]

# Define grid size (adjust for the number of nodes)
n_rows = 2
n_cols = 5

target = "mean"  # Choose between 'mean', 'std_dev', 'variance'
metric = "mse"  # Choose between 'mse', 'mae', 'r2'

# Initialize subplots with a grid layout
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

# Extract and plot data for each selected node
for idx, node_id in enumerate(nodes_to_plot):
    if node_id not in data:
        continue

    # Extract data for 'data'
    splits = []
    mse_values = []
    for split, metrics in data.get(node_id, {}).get(target, {}).items():
        splits.append(float(split))
        mse_values.append(metrics[metric])

    # Extract data for 'merged_data'
    splits_merged = []
    mse_values_merged = []
    for split, metrics in merged_data.get(node_id, {}).get(target, {}).items():
        splits_merged.append(float(split))
        mse_values_merged.append(metrics[metric])

    # Extract data for 'naive_data'
    splits_naive = []
    mse_values_naive = []
    for split, metrics in naive_data.get(node_id, {}).get(target, {}).items():
        splits_naive.append(float(split))
        mse_values_naive.append(metrics[metric])

    # Sort data for better visual alignment
    sorted_indices = sorted(range(len(splits)), key=lambda i: splits[i])
    splits = [splits[i] for i in sorted_indices]
    mse_values = [mse_values[i] for i in sorted_indices]

    sorted_indices_merged = sorted(range(len(splits_merged)), key=lambda i: splits_merged[i])
    splits_merged = [splits_merged[i] for i in sorted_indices_merged]
    mse_values_merged = [mse_values_merged[i] for i in sorted_indices_merged]

    sorted_indices_naive = sorted(range(len(splits_naive)), key=lambda i: splits_naive[i])
    splits_naive = [splits_naive[i] for i in sorted_indices_naive]
    mse_values_naive = [mse_values_naive[i] for i in sorted_indices_naive]

    # Plot the data as lines
    axes[idx].plot(splits, mse_values, label="Single-link Regression", marker='o', color="blue", alpha=0.7)
    axes[idx].plot(splits_merged, mse_values_merged, label="Global Regression", marker='o', color="orange", alpha=0.7)
    axes[idx].plot(splits_naive, mse_values_naive, label="Single-link k-NN", marker='o', color="green", alpha=0.7)

    # Add grid, title, and adjust tick rotation
    axes[idx].set_title(f'Node {node_id}')
    axes[idx].grid(True)
    axes[idx].tick_params(axis='x', rotation=45)

    # Add legend to the first plot
    if idx == 0:
        axes[idx].legend(loc="upper left")

# Remove empty subplots if any
for i in range(len(nodes_to_plot), n_rows * n_cols):
    fig.delaxes(axes[i])

# Add common labels
fig.text(0.5, 0.04, 'Split Ratios', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'MSE', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.suptitle(f'Line Plots of MSE for Different Nodes for {target}', fontsize=16)
plt.show()
