import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import json # type: ignore
import pandas as pd # type: ignore

with open('../data/same-config-latency-series.json', 'r') as file:
    latency_stats_plot = json.load(file)

# Choose configurations to plot
# Replace these tuples with actual configurations from your data
# Get the ten first configurations from the latency_stats dictionary
configurations = list(latency_stats_plot.values())[0].keys()
selected_configurations = list(configurations)[:10]

selected_nodes = ['102', '105', '106']  # Replace with actual node IDs of interest

node_config_latencies = latency_stats_plot
data = []

# Prepare the data for seaborn
for node in selected_nodes:
    for config in selected_configurations:
        config_str = str(config)
        if node in node_config_latencies and config_str in node_config_latencies[node]:
            for latency in node_config_latencies[node][config_str]:
                data.append({
                    'Node': node,
                    'Configuration': config_str,
                    'Latency': latency
                })

# Convert data to DataFrame for easier plotting
df = pd.DataFrame(data)

# Plotting
num_nodes = len(selected_nodes)
fig, axes = plt.subplots(num_nodes, 1, figsize=(10, 6 * num_nodes), sharex=True)
fig.suptitle("Latency Distributions for Selected Nodes and Configurations", fontsize=16)

# Generate a violin plot for each node
for i, node in enumerate(selected_nodes):
    ax = axes[i]
    sns.violinplot(
        data=df[df['Node'] == node],
        x="Configuration",
        y="Latency",
        ax=ax,
        palette="muted"
    )
    ax.set_title(f"Node {node}")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True)

# Display the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()