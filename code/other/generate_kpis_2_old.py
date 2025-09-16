# This script corresponds to the new version of the logs, where we have the ID at the sending, reception and the number of transmissions (csma ok)

import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# File paths
file_title = 'diff-config-grenoble-1'
filename = '../data/' + file_title + '.txt'
output_json_file = '../data/' + file_title + '-transmissions-per-config.json'
output_json_file_stats = '../data/' + file_title + '-transmissions-per-config-stats.json'

# Initialize structures
current_config = {}
received_packets = set()  # Set to store received packet IDs
sent_packets = defaultdict(list)  # To track sent packets per node
node_transmissions = defaultdict(lambda: defaultdict(list))  # Per node and configuration

# Regular expressions for parsing
config_re = r"(\d+\.\d+);m3-(\d+);The configuration has been changed to: CSMA_MIN_BE=(\d+), CSMA_MAX_BE=(\d+), CSMA_MAX_BACKOFF=(\d+), FRAME_RETRIES=(\d+)"
send_packet_re = r"(\d+\.\d+);m3-(\d+);Sending packet content: (\d+)"
receive_packet_re = r"(\d+\.\d+);m3-(\d+);Data received from .* '(\d+)'"
csma_ok_re = r"(\d+\.\d+);m3-(\d+);csma ok: (\d+) for packet: (\d+)"

# Read and parse logs
with open(filename, 'r') as file:
    logs = file.readlines()

# Step 1: Parse received packets and store their IDs
for line in logs:
    receive_match = re.match(receive_packet_re, line)
    if receive_match:
        _, _, packet_id = receive_match.groups()
        received_packets.add(packet_id)

# Step 2: Parse configurations, sent packets, and CSMA transmissions
packet_transmissions = {}  # Tracks packet ID to transmission count
for line in logs:
    # Handle configuration changes
    config_match = re.match(config_re, line)
    if config_match:
        _, node, min_be, max_be, max_backoff, retries = config_match.groups()
        current_config[node] = (int(min_be), int(max_be), int(max_backoff), int(retries))
        continue

    # Handle sending packets
    send_match = re.match(send_packet_re, line)
    if send_match:
        _, node, packet_id = send_match.groups()
        sent_packets[node].append((packet_id, current_config.get(node)))
        # Default to 0 transmissions if not received
        packet_transmissions[packet_id] = 0
        continue

    # Handle CSMA attempts
    csma_match = re.match(csma_ok_re, line)
    if csma_match:
        _, node, attempts, packet_id = csma_match.groups()
        if packet_id in packet_transmissions:
            packet_transmissions[packet_id] = 1/int(attempts)  # Update transmissions

# Step 3: Build the series per configuration for each node
for node, packets in sent_packets.items():
    for packet_id, config in packets:
        transmissions = 0  # Default to 0 transmissions if not received
        if packet_id in received_packets:
            transmissions = packet_transmissions.get(packet_id, 0)  # Get transmissions for received packet
        
        # Append the transmission count for the corresponding node and configuration
        if config:
            node_transmissions[node][config].append(transmissions)

# Convert the data to a JSON-serializable format
output_data = {
    node: {str(config): series for config, series in config_dict.items()}
    for node, config_dict in node_transmissions.items()
}

# Write JSON output to file
with open(output_json_file, 'w') as outfile:
    json.dump(output_data, outfile)

output_data_stats = {}
for node, config_dict in output_data.items():
    res = {}
    # Calculate the mean, std_dev and variance for each configuration
    for config, series in config_dict.items():
        mean = sum(series) / len(series)
        print(node, config, len(series), mean)
        std_dev = (sum([(x - mean) ** 2 for x in series]) / len(series)) ** 0.5
        variance = std_dev ** 2
        res[config] = {
            'mean': mean,
            'std_dev': std_dev,
            'variance': variance
        }
    output_data_stats[node] = res

# Write JSON output to file
with open(output_json_file_stats, 'w') as outfile:
    json.dump(output_data_stats, outfile, indent=4)

#print(output_data)
#print(output_data_stats)

data = output_data_stats

# Sort the nodes by ID
data = dict(sorted(data.items(), key=lambda x: int(x[0])))

metrics = ["mean", "std_dev"]

# Create subplots
num_nodes = len(data)
fig, axes = plt.subplots(num_nodes, 1, figsize=(10, 6 * num_nodes), sharex=False)

# Handle single subplot case
if num_nodes == 1:
    axes = [axes]

for ax, (node_id, tuples_data) in zip(axes, data.items()):
    tuples = list(tuples_data.keys())
    values = {metric: [tuples_data[t][metric] for t in tuples] for metric in metrics}
    x = np.arange(len(tuples))  # Tuple positions
    width = 0.2  # Bar width

    # Plot bars
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width, label=metric)

    # Formatting
    ax.set_title(f"Metrics for Node {node_id}")
    ax.set_ylabel("Values")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tuples, rotation=45, ha='right')
    ax.legend()

# Add shared labels and adjust layout
plt.xlabel("Tuples")
plt.tight_layout()
#plt.show()

# Prepare data for boxplots
nodes = list(data.keys())
metrics = ["mean"]
boxplot_data = []

for node_id in nodes:
    node_metrics = {metric: [] for metric in metrics}
    for tuple_key, values in data[node_id].items():
        for metric in metrics:
            node_metrics[metric].append(values[metric])
    boxplot_data.append([node_metrics[metric] for metric in metrics])

# Flatten data for plotting
flattened_data = [item for sublist in boxplot_data for item in sublist]
flattened_labels = [f"{node}-{metric}" for node in nodes for metric in metrics]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
positions = np.arange(len(flattened_data)) + 1
ax.violinplot(flattened_data, positions=positions, widths=0.6, showmedians=True)
#ax.boxplot(flattened_data, positions=positions, patch_artist=True, showfliers=True)

# Add custom x-tick labels for each group
group_positions = np.arange(1.5, len(flattened_data) + 1, len(metrics))
ax.set_xticks(group_positions)
ax.set_xticklabels(nodes)

# Formatting
ax.set_title("Boxplots of Metrics for All Nodes")
ax.set_ylabel("Values")
ax.set_xlabel("Nodes")
plt.xticks(rotation=45, ha="right")
ax.grid(axis="y")

# Add group labels for metrics
for i, node_id in enumerate(nodes):
    for j, metric in enumerate(metrics):
        x_pos = i * len(metrics) + j + 1
        ax.text(x_pos, ax.get_ylim()[1] * 0.95, metric, ha='center', fontsize=8)

plt.tight_layout()
#plt.show()


data = output_data

# Prepare data for plotting
node_ids = data.keys()
fig, axes = plt.subplots(len(node_ids), 1, figsize=(10, 5 * len(node_ids)), sharex=True)

if len(node_ids) == 1:
    axes = [axes]  # Ensure axes is a list for consistent handling

for ax, node_id in zip(axes, node_ids):
    node_data = data[node_id]
    concatenated_series = []
    change_indices = []
    start_idx = 0
    
    for config, series in node_data.items():
        concatenated_series.extend(series)
        start_idx += len(series)
        change_indices.append(start_idx - 1)  # Configuration change at the end of the series

    # Plot the data
    ax.plot(range(len(concatenated_series)), concatenated_series, label=f'Node {node_id}')
    
    # Add vertical lines for configuration changes
    for change_idx in change_indices[:-1]:  # Ignore last as it is end of the series
        ax.axvline(x=change_idx, color='red', linestyle='--', alpha=0.7, label='Config Change')

    ax.set_title(f'Evolution of Transmissions for Node {node_id}')
    ax.set_ylabel('Number of Transmissions')
    ax.grid()

# Global x-label
plt.xlabel('Time (Index)')
plt.tight_layout()
#plt.show()