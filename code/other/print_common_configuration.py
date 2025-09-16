import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from collections import Counter
import seaborn as sns # type: ignore

# Load sample JSON Data
with open('../data/diff-config-long-1-transmissions-per-config-series.json', 'r') as file:
    data = json.load(file)

sns.set(style="whitegrid")
sns.set(font_scale=1.4)

# Step 1: Extract configurations and identify the most common one
configs = []

for node, node_data in data.items():
    for config in node_data.keys():
        configs.append(config)

config_counter = Counter(configs)
most_common_configs = config_counter.most_common()
target_config = most_common_configs[0][0]  # Most common configuration
print(f"Target configuration: {target_config}")

# Step 2: Collect data for the target configuration
config_data = {}
for node, node_data in data.items():
    if target_config in node_data:
        config_data[node] = node_data[target_config]

# Step 3: Sort nodes lexicographically
sorted_nodes = sorted(config_data.keys())
sorted_nodes.remove('m3-153')  # Remove node m3-153

# Step 4: Compute global maximum to define bin edges
max_ = 0
hist_data = {}

for node in sorted_nodes:
    # Calculate transmission counts for the selected configuration
    transmission_counts = [int(1 / e) if e != 0 else 0 for e in config_data[node][:100]] # Truncate to 100 transmissions
    #print(transmission_counts, len(transmission_counts))
    #print(node)
    #print("Number of 0: ", transmission_counts.count(0))
    #print("Number of 1: ", transmission_counts.count(1))
    #print("Number of 2: ", transmission_counts.count(2))
    #print("Number of 3: ", transmission_counts.count(3))
    #print("Number of 4: ", transmission_counts.count(4))
    #print("Number of 5: ", transmission_counts.count(5))
    #print("Number of 6: ", transmission_counts.count(6))
    #print("Number of 7: ", transmission_counts.count(7))
    hist_data[node] = transmission_counts
    local_max = np.max(transmission_counts)
    if local_max > max_:
        max_ = local_max

bin_edges = np.linspace(0, max_, max_ + 1)

# Step 5: Prepare data for per-node stacked histograms
node_positions = np.arange(len(sorted_nodes))  # x-axis positions for nodes
bar_width = 0.8  # Width of each stacked bar

plt.figure(figsize=(12, 6))

# Initialize "bottom" for stacking at each node position
bottoms = np.zeros(len(sorted_nodes))  # One bottom value per node

# Each bin contributes to the stack
for i in range(1, len(bin_edges)):
    # Heights at this bin for all nodes
    bin_heights = [hist_data[node].count(i) for node in sorted_nodes]
    plt.bar(
        node_positions, bin_heights, bar_width,
        bottom=bottoms, label=f"{i}", 
    )
    bottoms += bin_heights  # Increment bottom for stacking

bin_heights = [hist_data[node].count(0) for node in sorted_nodes]
plt.bar(
    node_positions, bin_heights, bar_width,
    bottom=bottoms, label="Failures", 
)
bottoms += bin_heights  # Increment bottom for stacking

# Step 6: Plot formatting
plt.title(f"Transmission Frequencies for Configuration: {target_config}")
plt.xlabel("Nodes")
plt.ylabel("Frequency")
plt.xticks(node_positions, sorted_nodes, rotation=45)  # Node labels on x-axis
plt.ylim(0, 102)
plt.legend(title="# Transmissions", loc='upper right', fontsize='small')
plt.tight_layout()

# Save and show the plot
plt.savefig("../figures/stacked_histogram_per_node_8_8_5_2.pdf", format="pdf", bbox_inches="tight")
plt.show()
