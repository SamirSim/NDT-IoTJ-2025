import matplotlib.pyplot as plt  # type: ignore
import json  # type: ignore

file_title = 'diff-config'

# Load data from JSON files
with open(f'../data/{file_title}-results-split-multioutput.json', 'r') as file:
    data = json.load(file)

with open(f'../data/{file_title}-merged-results-split-multioutput.json', 'r') as file:
    merged_data = json.load(file)

# Nodes to plot
nodes_to_plot = ["101", "102", "103", "104", "105", "106", "108", "109", "110"]

# Define grid size (2 rows x 5 columns for 10 nodes)
n_rows = 2
n_cols = 5

target = "mse"  # Choose between 'mse', 'mae', 'r2'

# Initialize subplots with a grid layout
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

# Extract and plot data for each selected node
for idx, node_id in enumerate(nodes_to_plot):
    splits = []
    target_values = []
    splits_merged = []
    target_values_merged = []

    # Extract data from the 'data' dictionary for the original configuration
    for split, metrics in data.get(node_id, {}).items():
        splits.append(float(split))
        target_values.append(metrics[target][0])  # Extract the target metric (e.g., mse)

    # Sort splits and target_values for consistency in the plot
    sorted_indices = sorted(range(len(splits)), key=lambda i: splits[i])
    splits = [splits[i] for i in sorted_indices]
    target_values = [target_values[i] for i in sorted_indices]

    # Extract data from the 'merged_data' dictionary
    for split, metrics in merged_data.get(node_id, {}).items():
        splits_merged.append(float(split))
        target_values_merged.append(metrics[target][0])  # Extract the target metric (e.g., mse)

    # Sort splits_merged and target_values_merged
    sorted_indices_merged = sorted(range(len(splits_merged)), key=lambda i: splits_merged[i])
    splits_merged = [splits_merged[i] for i in sorted_indices_merged]
    target_values_merged = [target_values_merged[i] for i in sorted_indices_merged]

    # Plot on the corresponding subplot
    axes[idx].plot(splits, target_values, marker='o', label='Original', color='blue')
    axes[idx].plot(splits_merged, target_values_merged, marker='o', label='Merged', color='orange')
    axes[idx].set_title(f'Node {node_id}')
    axes[idx].grid(True)
    axes[idx].legend()

# Remove empty subplots if any
for i in range(len(nodes_to_plot), n_rows * n_cols):
    fig.delaxes(axes[i])

# Add common labels
fig.text(0.5, 0.04, 'Split Ratio', ha='center', fontsize=12)
fig.text(0.04, 0.5, target.upper(), va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.suptitle(f'Evolution of {target.upper()} for Different Nodes', fontsize=16)
plt.show()
