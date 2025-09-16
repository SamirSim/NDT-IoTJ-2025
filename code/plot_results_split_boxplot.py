import matplotlib.pyplot as plt # type: ignore
import json
import numpy as np # type: ignore
import time
import seaborn as sns # type: ignore

sns.set(font_scale=2.4)

#plt.rcParams['font.family'] = 'Helvetica'

file_title = 'diff-config-long'
# Load data from JSON files
with open('../data/' + file_title + '-results-mul-runs-split.json', 'r') as file: # Default split is 0.3
    data = json.load(file)

with open('../data/' + file_title + '-merged-results-mul-runs-split.json', 'r') as file:
    merged_data = json.load(file)

with open('../data/' + file_title + '-naive-results-mul-runs-split.json', 'r') as file:
    naive_data = json.load(file)

# Nodes to plot
#nodes_to_plot = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
nodes_to_plot = ["m3-123", "m3-133", "m3-143", "m3-159", "m3-163", "m3-166"]
#nodes_to_plot = ["m3-123"]

# Define grid size (2 rows x 5 columns for 10 nodes)
n_rows = 3
n_cols = 2

target = "mean"  # Choose between 'mean', 'std_dev', 'variance'
metric = "mse"  # Choose between 'mse', 'mae', 'r2'

# Initialize subplots with a grid layout
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 18), sharex=False, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

# Extract and plot data for each selected node
for idx, node_id in enumerate(nodes_to_plot):
    if node_id not in data:
        continue
    splits = []
    mse_values = []
    splits_merged = []
    mse_values_merged = []
    splits_naive = []
    mse_values_naive = []

    # Extract data from the 'data' dictionary
    for split, metrics in data.get(node_id, {}).get(target, {}).items():
        splits.append(float(split))
        mse_values.append(metrics[metric])  # Collect all values for boxplots

    for split in splits:
        print(f"Node {node_id} Split {split} Original MSE: {mse_values} Size: {len(mse_values)}")
    #time.sleep(2)

    # Extract data from the 'merged_data' dictionary
    for split, metrics in merged_data.get(node_id, {}).get(target, {}).items():
        splits_merged.append(float(split))
        mse_values_merged.append(metrics[metric])  # Collect all values for boxplots

    for split, metrics in naive_data.get(node_id, {}).get(target, {}).items():
        splits_naive.append(float(split))
        mse_values_naive.append(metrics[metric])

    # Ensure splits and their values are sorted for both approaches
    sorted_indices = sorted(range(len(splits)), key=lambda i: splits[i])
    splits = [splits[i] for i in sorted_indices]
    mse_values = [mse_values[i] for i in sorted_indices]

    sorted_indices_merged = sorted(range(len(splits_merged)), key=lambda i: splits_merged[i])
    splits_merged = [splits_merged[i] for i in sorted_indices_merged]
    mse_values_merged = [mse_values_merged[i] for i in sorted_indices_merged]

    sorted_indices_naive = sorted(range(len(splits_naive)), key=lambda i: splits_naive[i])
    splits_naive = [splits_naive[i] for i in sorted_indices_naive]
    mse_values_naive = [mse_values_naive[i] for i in sorted_indices_naive]

    # Prepare positions for side-by-side boxplots
    box_positions = np.arange(len(splits)) * 3  # Space for Original and Merged pairs
    box_positions_merged = box_positions + 0.8  # Offset for "Merged" data
    box_positions_naive = box_positions + 1.6  # Offset for "Naive" data

    # Plot Original, Merged and Naive boxplots
    axes[idx].boxplot(
        mse_values, positions=box_positions, widths=0.6, patch_artist=True, 
        boxprops=dict(facecolor="blue", alpha=0.5)
    )
    axes[idx].boxplot(
        mse_values_merged, positions=box_positions_merged, widths=0.6, patch_artist=True, 
        boxprops=dict(facecolor="orange", alpha=0.5)
    )
    axes[idx].boxplot(
        mse_values_naive, positions=box_positions + 1.6, widths=0.6, patch_artist=True, 
        boxprops=dict(facecolor="green", alpha=0.5)
    )

    # Set x-axis labels at the midpoint between paired boxplots
    mid_positions = (box_positions + box_positions_naive) / 2
    axes[idx].set_xticks(mid_positions)
    axes[idx].set_xticklabels([f"{split:.2f}" for split in splits])

    # Title and grid
    axes[idx].set_title(f'Link {node_id}')
    axes[idx].grid(True)
    axes[idx].tick_params(axis='x', rotation=45)

    # Corrected legend
    if idx == 0:
        axes[idx].legend(
            handles=[
                plt.Line2D([0], [0],lw=4, alpha=0.5, label="Single-link Regression"),
                plt.Line2D([0], [0], color="orange", lw=4, alpha=0.5, label="Global Regression"),
                plt.Line2D([0], [0], color="green", lw=4, alpha=0.5, label="Single-link k-NN")
            ],
            loc="upper left"
        )

# Remove empty subplots if any
for i in range(len(nodes_to_plot), n_rows * n_cols):
    fig.delaxes(axes[i])

# Add common labels
fig.text(0.5, 0.04, 'Split Ratios', ha='center', fontsize=14)
#fig.text(0.04, 0.5, 'MSE', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

#plt.suptitle(f'Boxplots of MSE for Different Nodes for {target} reception ratio ', fontsize=30)
plt.savefig(f'../figures/results-split-boxplot-2.pdf', format='pdf', bbox_inches='tight')
plt.show()
