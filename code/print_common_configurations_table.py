import pandas as pd # type: ignore
import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import random

sns.set(font_scale=1.2)
#plt.rcParams['font.family'] = 'Helvetica'

# Example input data
data = {
    "m3-163": {
        "(3, 8, 5, 5)": {"mean": 0.33, "std_dev": 0.448441746495573, "variance": 0.20109999999999975},
        "(6, 7, 4, 7)": {"mean": 0.16075000000000003, "std_dev": 0.3350120126370273, "variance": 0.11223304861111173},
        "(4, 4, 3, 5)": {"mean": 0.26816666666666666, "std_dev": 0.3974718368779241, "variance": 0.1579838611111111},
        "(7, 10, 1, 2)": {"mean": 0.47666666666666674, "std_dev": 0.4434836837785326, "variance": 0.19667777777777748},
    },
    # Add more nodes and configurations as needed
    "m3-166": {
        "(3, 8, 5, 5)": {"mean": 0.28, "std_dev": 0.41, "variance": 0.1681},
        "(4, 4, 3, 5)": {"mean": 0.26, "std_dev": 0.39, "variance": 0.1521},
        "(7, 10, 1, 2)": {"mean": 0.48, "std_dev": 0.44, "variance": 0.1936},
    }
}

random.seed(40)

with open ('../data/AINA/diff-config-long-1-transmissions-per-config-stats.json', 'r') as file:
    data = json.load(file)

# Identify all configurations that are shared by at least two nodes
config_counts = {}
for node_data in data.values():
    for config in node_data.keys():
        config_counts[config] = config_counts.get(config, 0) + 1

common_configurations = [config for config, count in config_counts.items() if count >= 6]

# Check if we have at least 10 common configurations
if len(common_configurations) < 10:
    print("Not enough configurations shared by at least two nodes!")
    exit()

# Randomly choose 10 common configurations and 5 nodes
selected_configurations = random.sample(common_configurations, 15)
selected_nodes = list(data.keys())

# Remove the node m3-153 if it exists
if "m3-153" in selected_nodes:
    pass
    #selected_nodes.remove("m3-153")

# Sort the nodes lexicographically
selected_nodes.sort()

# Initialize the DataFrame with "N/A" as the default value
table = pd.DataFrame("N/A", index=selected_nodes, columns=sorted(selected_configurations))

# Populate the DataFrame with mean values
for node in selected_nodes:
    for config in selected_configurations:
        if config in data[node]:
            table.loc[node, config] = round(data[node][config]["mean"], 2)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    table.replace("N/A", None).astype(float),  # Use NaN for plotting, "N/A" for display
    annot=table,  # Annotate the heatmap with the table content (numbers and "N/A")
    fmt="",  # Prevent seaborn from formatting annotations
    cmap="RdYlGn",  # Colormap
    cbar_kws={'label': 'Mean Reception Rate'},  # Label for the color bar
    linewidths=0.5
)

# Adjust labels
plt.title("Mean Reception Rate for Common Configurations")
plt.xlabel("Configurations")
plt.ylabel("Links")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("../figures/common_configurations_table.pdf", format="pdf", bbox_inches="tight")

# Show the plot
plt.show()