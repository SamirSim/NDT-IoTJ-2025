import matplotlib.pyplot as plt  # type: ignore
import json  # type: ignore
from time import sleep

file_title = 'diff-config-transmissions-per-config'

# Load data from JSON files
with open('../data/'+file_title+'-results-split.json', 'r') as file:
    data = json.load(file)

with open('../data/'+file_title+'-merged-results-split.json', 'r') as file:
    merged_data = json.load(file)

# Nodes to plot
#nodes_to_plot = ["101", "102", "103", "104", "105", "106", "108", "109", "110"]
#nodes_to_plot = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
#nodes_to_plot = ["102", "103", "106", "107", "110", "111"]
nodes_to_plot = ["123", "133", "143", "150", "159", "163", "166"]

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
    mse_values = []
    mse_values_merged = []

    # Extract data from the 'data' dictionary
    for split, metrics in data.get(node_id, {}).get(target, {}).items():
        splits.append(float(split))
        print(metrics["mse"])
        #sleep(2)
        mse_values.append([v for v in metrics["mse"]])

    # Sort splits and mse_values for consistency in the plot
    sorted_indices = sorted(range(len(splits)), key=lambda i: splits[i])
    splits = [splits[i] for i in sorted_indices]
    mse_values = [mse_values[i] for i in sorted_indices]

    # Extract data from the 'merged_data' dictionary
    splits_merged = []
    for split, metrics in merged_data.get(node_id, {}).get(target, {}).items():
        splits_merged.append(float(split))
        mse_values_merged.append([v for v in metrics["mse"]])

    sorted_indices_merged = sorted(range(len(splits_merged)), key=lambda i: splits_merged[i])
    splits_merged = [splits_merged[i] for i in sorted_indices_merged]
    mse_values_merged = [mse_values_merged[i] for i in sorted_indices_merged]

    # Plot on the corresponding subplot
    axes[idx].plot(splits, mse_values, marker='o', label='Original', color='blue')
    axes[idx].plot(splits_merged, mse_values_merged, marker='x', label='Merged', color='orange')
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
