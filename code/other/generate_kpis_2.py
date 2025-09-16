import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import time

# File paths
file_titles = [
    "diff-config-grenoble",
    "diff-config-grenoble-2",
    "diff-config-grenoble-3",
    "diff-config-grenoble-4",
]
file_paths = [f"../data/{title}.txt" for title in file_titles]
output_json_file = "../data/diff-config-transmissions-per-config-series.json"
output_json_file_stats = "../data/diff-config-transmissions-per-config-stats.json"

node_transmissions = defaultdict(lambda: defaultdict(list))  # Per node and configuration

# Regular expressions for parsing
config_re = r"(\d+\.\d+);m3-(\d+);The configuration has been changed to: CSMA_MIN_BE=(\d+), CSMA_MAX_BE=(\d+), CSMA_MAX_BACKOFF=(\d+), FRAME_RETRIES=(\d+)"
send_packet_re = r"(\d+\.\d+);m3-(\d+);Sending packet content: (\d+)"
receive_packet_re = r"(\d+\.\d+);m3-(\d+);Data received from .* '(\d+)'"
csma_ok_re = r"(\d+\.\d+);m3-(\d+);csma ok: (\d+) for packet: (\d+)"

# Read and parse logs from all files
for filename in file_paths:

    # Initialize structures
    current_config = {}
    received_packets = set()  # Set to store received packet IDs
    sent_packets = defaultdict(list)  # To track sent packets per node
    
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
            # Default to 100 transmissions if not received
            packet_transmissions[packet_id] = 0
            continue

        # Handle CSMA attempts
        csma_match = re.match(csma_ok_re, line)
        if csma_match:
            _, node, attempts, packet_id = csma_match.groups()
            if packet_id in packet_transmissions:
                packet_transmissions[packet_id] = 1 / int(attempts)  # Update transmissions

    # Step 3: Build the series per configuration for each node
    for node, packets in sent_packets.items():
        for packet_id, config in packets:
            transmissions = 0  # Default to 0 transmissions if not received
            if packet_id in received_packets:
                transmissions = packet_transmissions.get(packet_id, 0)  # Get transmissions for received packet

            # Append the transmission count for the corresponding node and configuration
            if config:
                node_transmissions[node][config].append(transmissions)
    #print(node_transmissions["153"][(7, 9, 1, 2)])
    #print("\n")
    #time.sleep(2)

# Convert the data to a JSON-serializable format
output_data = {
    node: {str(config): series for config, series in config_dict.items()}
    for node, config_dict in node_transmissions.items()
}

# Write JSON output to file
with open(output_json_file, 'w') as outfile:
    json.dump(output_data, outfile)

# Compute and store statistics
output_data_stats = {}
for node, config_dict in output_data.items():
    res = {}
    # Calculate the mean, std_dev, and variance for each configuration
    for config, series in config_dict.items():
        #print(f"Node: {node}, Config: {config}, Series: {series}")
        #time.sleep(2)
        if len(series) > 0:
            mean = sum(series) / len(series)
            std_dev = (sum([(x - mean) ** 2 for x in series]) / len(series)) ** 0.5
            variance = std_dev ** 2
            res[config] = {
                "mean": mean,
                "std_dev": std_dev,
                "variance": variance,
            }
    output_data_stats[node] = res

# Write JSON output to file
with open(output_json_file_stats, 'w') as outfile:
    json.dump(output_data_stats, outfile, indent=4)

# Combine data and plot
data = output_data_stats

# Sort the nodes by ID
data = dict(sorted(data.items(), key=lambda x: int(x[0])))

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
ax.violinplot(flattened_data, positions=positions, widths=0.6, showmeans=True, showextrema=True)
#ax.boxplot(flattened_data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor="blue", alpha=0.5))

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

plt.tight_layout()
plt.show()
