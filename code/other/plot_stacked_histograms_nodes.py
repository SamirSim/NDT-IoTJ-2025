import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore

sns.set(style="whitegrid")

# Input JSON file
input_json_file = "../data/AINA/diff-config-long-1-transmissions-per-config-series.json"

# Load data from JSON file
with open(input_json_file, 'r') as infile:
    data = json.load(infile)

# Prepare data for plotting
node_ids = sorted(data.keys(), key=lambda x: int(x.split('-')[1]))

# Collect unique transmission counts and their frequencies for each node
node_frequencies = {}
for node in node_ids:
    node_data = data[node]
    transmissions = []
    
    for config in node_data.values():
        if len(config) < 90:
            print(config, len(config))
        else:
            config_ = [0 if e == 0 else int(1/e) for e in config]
            transmissions.extend(config_)
    unique, counts = np.unique(transmissions, return_counts=True)
    
    # Combine transmissions greater than 0.25 into one category
    combined_freqs = {}
    for u, c in zip(unique, counts):
        combined_freqs[u] = c

    node_frequencies[node] = combined_freqs

print(node_frequencies)

data = node_frequencies
# Step 1: Extract nodes and bins
nodes = sorted(data.keys())
# Remove the node m3-153
#nodes.remove('m3-153')
bins = sorted(set(key for node in data.values() for key in node.keys()))  # Bin labels (0, 1, ..., 8)

# Step 2: Prepare data for stacked bar plot
node_positions = np.arange(len(nodes))  # x-axis positions
bar_width = 0.8  # Width of each bar
colors = plt.cm.tab20(range(len(bins)))  # Use colormap for distinct bin colors

# Initialize bottom heights for stacking
bottoms = np.zeros(len(nodes))

# Step 3: Plot the stacked bar plot
#plt.figure()

for i, bin_ in enumerate(bins):
    if i != 0:
        # Heights for the current bin across nodes
        heights = [data[node].get(bin_, 0) for node in nodes]
        plt.bar(node_positions, heights, width=bar_width, bottom=bottoms, color=colors[i],
                edgecolor='black', label=f'{bin_}')
        # Update bottoms for stacking
        bottoms += heights
    else:
        bin_0 = bin_

heights = [data[node].get(bin_0, 0) for node in nodes]
plt.bar(node_positions, heights, width=bar_width, bottom=bottoms, color=colors[0],
        edgecolor='black', label='Failures')
# Update bottoms for stacking
bottoms += heights

# Step 4: Formatting
plt.title("Transmission Counts Per Node")
plt.xlabel("Nodes")
plt.ylabel("Frequency of Transmission Counts")
plt.xticks(node_positions, nodes, rotation=45)
plt.legend(title="# Transmissions", loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')
plt.tight_layout()

# Step 5: Show and Save
plt.savefig("../figures/stacked_histogram_nodes_bins.pdf", format="pdf", bbox_inches="tight")
#plt.show()