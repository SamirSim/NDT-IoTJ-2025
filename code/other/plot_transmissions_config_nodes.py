import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import random
import time

sns.set(style="whitegrid")
random.seed(40)

sns.set(font_scale=1.4)

# File paths
file_titles = [
    "diff-config-long-1",
    "diff-config-long-2",
    "diff-config-long-3",
    "diff-config-long-4",
]
file_paths = [f"../data/{title}-transmissions-per-config-series.json" for title in file_titles]

# Load the data for selected nodes
input_file = file_paths[0]
with open(input_file, 'r') as file:
    node_transmissions = json.load(file)

# Nodes to display
selected_nodes = ['m3-123', 'm3-159', 'm3-166']  # Add your selected nodes here

# Number of configurations to randomly select for each node
num_random_configs = 10

# Create the subplots (3 rows, 1 column for 3 nodes)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

# Process each node
for i, node in enumerate(selected_nodes):
    node_data = node_transmissions[node]

    # Get the list of all configurations (sorted)
    configs = sorted(node_data.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
    selected_configs = random.sample(configs, min(num_random_configs, len(configs)))

    # Check if the selected configurations have 100 transmissions, otherwise generate new ones
    while any(len(node_data[config]) != 100 for config in selected_configs):
        selected_configs = random.sample(configs, min(num_random_configs, len(configs)))

    max_ = 0
    
    for config in selected_configs:
        # Compute the max number of transmissions for each configuration
        transmission_counts = [int(1 / e) if e != 0 else 0 for e in node_data[config]]
        max_transmissions = np.max(transmission_counts)
        if max_transmissions > max_:
            max_ = max_transmissions
    try:
        print("here: ", node, node_data["(4, 8, 8, 6)"], len(node_data["(4, 8, 8, 6)"]))
    except:
        pass

    if node == "m3-166":
        print(max_ )
    bin_edges = np.linspace(0, max_, max_+2)

    # Prepare data for the stacked bar plot
    data_dict = defaultdict(list)
    for config in selected_configs:
        # Filter and process transmission values
        if len(node_data[config]) != 100:
            continue  # Skip overly large configurations for clarity
        transmission_counts = [int(1 / e) if e != 0 else 0 for e in node_data[config]]
            
        print(config, transmission_counts)
        print(bin_edges)
       # time.sleep(1)
        binned_counts, _ = np.histogram(transmission_counts, bins=bin_edges)
        data_dict[config] = binned_counts
       
        print(config, binned_counts)

    colors = plt.cm.tab20(range(len(bin_edges)+1))  # Use colormap for distinct bin colors

    # Convert to a DataFrame for Seaborn plotting
    df = pd.DataFrame(data_dict).T
    df.index.name = "Configuration"
    df = df.reset_index()

    # Replace the name of the 0 column by "Failures"
    df.rename(columns={0: "Failures"}, inplace=True)

    # Place this column at the end
    cols = list(df.columns)
    cols.remove("Failures")
    cols.append("Failures")
    df = df[cols]

    print(df)

    # Plot the stacked bar chart in the appropriate subplot
    ax = axes[i]  # Select the correct subplot for the node
    if i < 2:
        df.set_index("Configuration").plot(
            kind="bar",
            stacked=True,
            ax=ax,  # Set the subplot for this plot
            width=0.3,
            color=colors,
        )

    # Customize subplot
    ax.set_title(f"Node {node}")
    ax.set_xlabel("")
    if i == 2:
        colors_166 = [colors[0], colors[1], colors[2], colors[3], colors[6]]
        ax.set_xlabel("Configurations")
        # Change the color of the Failures column

        df.set_index("Configuration").plot(
            kind="bar",
            stacked=True,
            ax=ax,  # Set the subplot for this plot
            width=0.3,
            color=colors_166,
        )
        
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="# Transmissions", loc="upper right", fontsize="small")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("../figures/stacked_bar_nodes.pdf", format="pdf", bbox_inches="tight")
plt.show()
