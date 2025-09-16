import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d

file_title = 'diff-config'
# Load data from JSON files
with open('../data/' + file_title + '-results-mul-runs-split.json', 'r') as file:
    data = json.load(file)

with open('../data/' + file_title + '-merged-results-mul-runs-split.json', 'r') as file:
    merged_data = json.load(file)

# Nodes to plot
nodes_to_plot = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

# Define grid size (2 rows x 5 columns for 10 nodes)
n_rows = 2
n_cols = 5

target = "mean"  # Choose between 'mean', 'std_dev', 'variance'

# Initialize subplots with a grid layout
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

# Extract and plot data for each selected node
for idx, node_id in enumerate(nodes_to_plot):
    splits = []
    mse_means = []
    mse_min = []
    mse_max = []
    mse_means_merged = []
    mse_min_merged = []
    mse_max_merged = []

    # Extract data from the 'data' dictionary
    for split, metrics in data.get(node_id, {}).get(target, {}).items():
        splits.append(float(split))
        mse_values = metrics["mse"]
        mse_means.append(sum(mse_values) / len(mse_values))
        mse_min.append(min(mse_values))
        mse_max.append(max(mse_values))

    # Skip nodes with no data
    if len(splits) < 2:
        axes[idx].set_title(f'Node {node_id} (No Data)')
        axes[idx].grid(True)
        continue

    # Sort splits and corresponding values for consistency in the plot
    sorted_indices = sorted(range(len(splits)), key=lambda i: splits[i])
    splits = np.array([splits[i] for i in sorted_indices])
    mse_means = np.array([mse_means[i] for i in sorted_indices])
    mse_min = np.array([mse_min[i] for i in sorted_indices])
    mse_max = np.array([mse_max[i] for i in sorted_indices])

    # Interpolate data for smoothing
    fine_splits = np.linspace(splits.min(), splits.max(), 300)
    mse_means_smooth = interp1d(splits, mse_means, kind='cubic')(fine_splits)
    mse_min_smooth = interp1d(splits, mse_min, kind='cubic')(fine_splits)
    mse_max_smooth = interp1d(splits, mse_max, kind='cubic')(fine_splits)

    # Extract data from the 'merged_data' dictionary
    splits_merged = []
    for split, metrics in merged_data.get(node_id, {}).get(target, {}).items():
        splits_merged.append(float(split))
        mse_values = metrics["mse"]
        mse_means_merged.append(sum(mse_values) / len(mse_values))
        mse_min_merged.append(min(mse_values))
        mse_max_merged.append(max(mse_values))

    if len(splits_merged) < 2:
        splits_merged = []
        mse_means_merged = []
        mse_min_merged = []
        mse_max_merged = []

    if splits_merged:
        sorted_indices_merged = sorted(range(len(splits_merged)), key=lambda i: splits_merged[i])
        splits_merged = np.array([splits_merged[i] for i in sorted_indices_merged])
        mse_means_merged = np.array([mse_means_merged[i] for i in sorted_indices_merged])
        mse_min_merged = np.array([mse_min_merged[i] for i in sorted_indices_merged])
        mse_max_merged = np.array([mse_max_merged[i] for i in sorted_indices_merged])

        # Interpolate data for smoothing
        fine_splits_merged = np.linspace(splits_merged.min(), splits_merged.max(), 300)
        mse_means_merged_smooth = interp1d(splits_merged, mse_means_merged, kind='cubic')(fine_splits_merged)
        mse_min_merged_smooth = interp1d(splits_merged, mse_min_merged, kind='cubic')(fine_splits_merged)
        mse_max_merged_smooth = interp1d(splits_merged, mse_max_merged, kind='cubic')(fine_splits_merged)

        # Plot mean with shaded area for Merged data
        axes[idx].fill_between(fine_splits_merged, mse_min_merged_smooth, mse_max_merged_smooth, color='orange', alpha=0.2, label='Merged Range')
        axes[idx].plot(fine_splits_merged, mse_means_merged_smooth, color='orange', label='Merged Mean')

    # Plot mean with shaded area for Original data
    axes[idx].fill_between(fine_splits, mse_min_smooth, mse_max_smooth, color='blue', alpha=0.2, label='Original Range')
    axes[idx].plot(fine_splits, mse_means_smooth, color='blue', label='Original Mean')

    axes[idx].set_title(f'Node {node_id}')
    axes[idx].grid(True)
    axes[idx].legend()

# Remove empty subplots if any
for i in range(len(nodes_to_plot), n_rows * n_cols):
    fig.delaxes(axes[i])

# Add common labels
fig.text(0.5, 0.04, 'Split Ratio', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'MSE', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.suptitle(f'Evolution of MSE for Different Nodes for {target}', fontsize=16)
plt.show()
