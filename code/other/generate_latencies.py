import re
import json
import numpy as np # type: ignore
from collections import defaultdict
import pandas as pd # type: ignore

# Helper function to convert tuple keys to strings
def convert_keys_to_strings(d):
    return {str(k): v for k, v in d.items()}

# Function to compute statistical data
def compute_statistics(data):
    if len(data) == 0:
        return {}

    stats = {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std_dev': float(np.std(data)),
        'variance': float(np.var(data)),
        '25_percentile': float(np.percentile(data, 25)),
        '50_percentile': float(np.percentile(data, 50)),  # Same as median
        '75_percentile': float(np.percentile(data, 75)),
        'sum': float(np.sum(data)),
        'count': float(len(data))
    }
    return stats

# Initialize data structures
current_config = {}
sent_packets = {}
node_latency_counts = defaultdict(lambda: defaultdict(list))
file_title = 'same-config'
filename = '../data/' + file_title + '.txt'

# Read the logs from the file
with open(filename, 'r') as file:
    logs = file.read()

# Regular expressions to capture the necessary fields
config_re = re.compile(r'(\d+\.\d+);m3-(\d+);The configuration has been changed to: CSMA_MIN_BE=(\d+), CSMA_MAX_BE=(\d+), CSMA_MAX_BACKOFF=(\d+), FRAME_RETRIES=(\d+)')
send_packet_re = re.compile(r'(\d+\.\d+);m3-(\d+);Sending packet content: (\d+)')
receive_packet_re = re.compile(r'(\d+\.\d+);m3-\d+;Data received from fd00::[\da-f]+ on port \d+ from port \d+ with length \d+: \'(\d+)\'')
    
# Parse each line in the log
for line in logs.splitlines():
    # Check for configuration change
    config_match = config_re.search(line)
    if config_match:
        timestamp, node, min_be, max_be, max_backoff, retries = config_match.groups()
        current_config[node] = (int(min_be), int(max_be), int(max_backoff), int(retries))

    # Check for packet sending
    send_match = send_packet_re.search(line)
    if send_match:
        timestamp, node, packet_id = send_match.groups()
        if node in current_config:
            sent_packets[packet_id] = (float(timestamp), node, current_config[node])

    # Check for packet reception
    receive_match = receive_packet_re.search(line)
    if receive_match:
        timestamp, packet_id = receive_match.groups()
        if packet_id in sent_packets:
            send_time, send_node, config = sent_packets[packet_id]
            latency = float(timestamp) - send_time
            # Append latency to the node's corresponding configuration
            node_latency_counts[send_node][config].append(latency)

# Convert tuple keys to strings and compute statistics
latency_stats = {}
latency_stats_plot = {}
for node, config_dict in node_latency_counts.items():
    node_stats = {}
    node_stats_plot = {}
    for config, latencies in config_dict.items():
        node_stats[str(config)] = compute_statistics(latencies)
        node_stats_plot[str(config)] = latencies
    latency_stats[node] = node_stats
    latency_stats_plot[node] = node_stats_plot

# Write JSON file for latency statistics
json_file = '../data/' + file_title + '-latency-stats.json'
with open(json_file, 'w') as outfile:
    json.dump(latency_stats, outfile, indent=4)

# Write JSON file for latency statistics
json_file = '../data/' + file_title + '-latency-series.json'
with open(json_file, 'w') as outfile:
    json.dump(latency_stats_plot, outfile, indent=4)

# Display the resulting statistics dictionary
print(latency_stats)

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

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