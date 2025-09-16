import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import random

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

# Load the data for the selected node
input_file = file_paths[0]
with open(input_file, 'r') as file:
    node_transmissions = json.load(file)

# Selected node to display
selected_node = 'm3-123'  # Focus on this node
node_data = node_transmissions[selected_node]

# Number of configurations to randomly select
num_random_configs = 10

# Get the list of all configurations (sorted)
configs = sorted(node_data.keys(), key=lambda x: tuple(map(int, re.findall(r'\d+', x))))
selected_configs = random.sample(configs, min(num_random_configs, len(configs)))

# Ensure selected configurations have valid data
while any(len(node_data[config]) != 100 for config in selected_configs):
    selected_configs = random.sample(configs, min(num_random_configs, len(configs)))

# Compute the max number of transmissions to determine bin edges
max_transmissions = 0
for config in selected_configs:
    transmission_counts = [int(1 / e) if e != 0 else 0 for e in node_data[config]]
    max_transmissions = max(max_transmissions, np.max(transmission_counts))

bin_edges = np.linspace(0, max_transmissions, max_transmissions + 2)

# Prepare data for the stacked bar plot
data_dict = defaultdict(list)
for config in selected_configs:
    if len(node_data[config]) != 100:
        continue  # Skip invalid configurations
    transmission_counts = [int(1 / e) if e != 0 else 0 for e in node_data[config]]
    binned_counts, _ = np.histogram(transmission_counts, bins=bin_edges)
    data_dict[config] = binned_counts

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

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab20(range(len(bin_edges) + 1))  # Use a colormap for bin colors
df.set_index("Configuration").plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.7,
    color=colors,
)

# Customize plot
ax.set_title(f"Transmission Distribution for Node {selected_node}")
ax.set_ylabel("Frequency")
ax.set_xlabel("Configuration")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(title="# Transmissions", loc="upper right", fontsize="small")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("../figures/stacked_bar_node_123.pdf", format="pdf", bbox_inches="tight")
plt.show()
