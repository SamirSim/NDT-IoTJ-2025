import re
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import defaultdict # type: ignore
import time
import json
import numpy as np # type: ignore

# Initialize a dictionary to store transmission counts for each node
transmission_counts = {}
file_title = 'diff-config-grenoble'
filename = '../data/'+file_title+'.txt'

# Read the logs from the file
with open(filename, 'r') as file:
    logs = file.read()

# Regular expressions to capture the necessary fields
config_re = r"m3-(\d+);The configuration has been changed to: CSMA_MIN_BE=(\d+), CSMA_MAX_BE=(\d+), CSMA_MAX_BACKOFF=(\d+), FRAME_RETRIES=(\d+)"
trans_count_re = r"m3-(\d+);csma: rexmit ok (\d+)"

# Patterns to capture timestamp, node ID, and the data for transmissions and config changes
transmission_pattern = r'(\d+\.\d+);m3-(\d+);csma: rexmit ok (\d+)'
config_pattern = r'(\d+\.\d+);m3-(\d+);The configuration has been changed to: (.*)'

# Dictionary to store transmission counts per node and configuration
node_transmission_counts = defaultdict(lambda: defaultdict(list))
transmission_timeline = defaultdict(list)

# Current configuration per node
current_config = {}

# Parsing the log data
for line in logs.splitlines():
    # Check for configuration change
    config_match = re.search(config_re, line)
    if config_match:
        node = config_match.group(1)
        csma_min_be = int(config_match.group(2))
        csma_max_be = int(config_match.group(3))
        csma_max_backoff = int(config_match.group(4))
        frame_retries = int(config_match.group(5))
        
        # Store the current configuration for the node
        current_config[node] = (csma_min_be, csma_max_be, csma_max_backoff, frame_retries)
        
        # Record the time of configuration change for the node
        timestamp = float(line.split(";")[0])
        transmission_timeline[node].append((timestamp, current_config[node]))

    # Check for transmission count
    trans_match = re.search(trans_count_re, line)
    if trans_match:
        node = trans_match.group(1)
        transmission_count = int(trans_match.group(2))
        
        # Get the current configuration for the node
        if node in current_config:
            config = current_config[node]
            # Append the transmission count to the respective configuration's list
            node_transmission_counts[node][config].append(transmission_count)
            
            # Record the timestamp and transmission count for plotting
            timestamp = float(line.split(";")[0])
            transmission_timeline[node].append((timestamp, transmission_count))


# Create a figure
#nodes = ["95", "101", "102", "103", "104", "105", "106", "108", "109"]
nodes = ["123", "133", "143", "150", "153", "159", "163", "166"]

print(node_transmission_counts["123"])
num_nodes = len(nodes)
plt.figure(figsize=(10, 2 * num_nodes))  # Adjust the figure height based on the number of nodes

to_plot = "boxplot"  # Choose between "boxplot" and "timeline"
#to_plot = "timeline"  # Choose between "boxplot" and "timeline"
#to_plot = "histogram"  # Choose between "boxplot" and "timeline"

# Plot the boxplots for each node
if to_plot == "boxplot":
    for i, node_id in enumerate(nodes):
        # Extract timestamps and transmission counts for the node
        timestamps = []
        transmissions = []
        config_changes = []
        
        for entry in transmission_timeline[node_id]:
            timestamp, value = entry
            timestamps.append(timestamp)
            if isinstance(value, tuple):  # Configuration change
                config_changes.append((timestamp, value))
            else:
                transmissions.append(value)
        
        # Generate for each configuration the succession of transmission counts
        config_transmissions = defaultdict(list)
        for timestamp, value in transmission_timeline[node_id]:
            if isinstance(value, tuple):
                config = value
            else:
                config_transmissions[config].append(value)

        # Prepare data for the boxplot
        data = [transmissions for transmissions in config_transmissions.values() if transmissions]

        min_bound = 0
        max_bound = 15
        # Add the configuration (8, 10, 3, 5) to the data
        data = data[min_bound:max_bound] # Limit the number of configurations for plotting
        if node_id == "123":
            for elem, value in node_transmission_counts[node_id].items():
                if elem == (8, 10, 3, 5):        
                    data.append(value)

        #data.append(node_transmission_counts[node_id]['(8, 10, 3, 5)'])
        config_transmissions = dict(list(config_transmissions.items())[min_bound:max_bound])
        if node_id == "123":
            for elem, value in node_transmission_counts[node_id].items():
                if elem == (8, 10, 3, 5):        
                    config_transmissions[(8, 10, 3, 5)] = value
        #config_transmissions['(8, 10, 3, 5)'] = node_transmission_counts[node_id]['(8, 10, 3, 5)']
        
        # Create a subplot for the current node
        plt.subplot(num_nodes, 1, i + 1)  # (rows, cols, panel number)
        
        # Create the boxplot for the current node
        plt.boxplot(data, patch_artist=True, showfliers=True)
        #plt.violinplot(data, showmedians=True)
        """
        mean = [np.mean(transmissions) for transmissions in data]
        variance = [np.var(transmissions) for transmissions in data]
        std_dev =[np.std(transmissions) for transmissions in data]
        plt.scatter(range(1, len(data) + 1), mean, label='Mean', marker='o', color='r')
        plt.scatter(range(1, len(data) + 1), variance, label='Variance', marker='o', color='g')
        plt.scatter(range(1, len(data) + 1), std_dev, label='Standard Deviation', marker='o', color='b')
        """
        # Remove xticks and labels
        #plt.xticks([])
        if i == 0:
            plt.legend()
        
        # Label each configuration on the x-axis
        plt.xticks(range(1, len(config_transmissions) + 1), [str(cfg) for cfg in config_transmissions.keys()], rotation=90)
        #plt.ylabel('Transmission Count')
        plt.title(f'Transmission Count Distribution for Node {node_id}')
        #plt.ylabel('Mean, Variance, Standard Deviation')
        #plt.title(f'Mean, Variance, and Standard Deviation of Transmissions for Node {node_id}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
elif to_plot == "timeline":
    # Plot the transmission counts over time for each node
    for i, node_id in enumerate(nodes):
        # Extract timestamps and transmission counts for the node
        timestamps = []
        transmissions = []
        config_changes = []
        
        for entry in transmission_timeline[node_id]:
            timestamp, value = entry
            timestamps.append(timestamp)
            if isinstance(value, tuple):  # Configuration change
                config_changes.append((timestamp, value))
            else:
                transmissions.append(value)
        
        # Generate for each configuration the succession of transmission counts
        config_transmissions = defaultdict(list)
        for timestamp, value in transmission_timeline[node_id]:
            if isinstance(value, tuple):
                config = value
            else:
                config_transmissions[config].append(value)
        
        # Create a subplot for the current node
        plt.subplot(num_nodes, 1, i + 1)  # (rows, cols, panel number)
        
        # Plot transmissions over time for the current node
        plt.plot(timestamps[:len(transmissions)], transmissions, label=f"Node {node_id}", marker='o')
        
        # Plot configuration changes as vertical lines
        for config_change in config_changes:
            ts, config = config_change
            plt.axvline(x=ts, color='r', linestyle='--', alpha=0.5)  # Adjust alpha for visibility
        
        # Customize each subplot
        plt.xlabel('Time (s)')
        plt.ylabel('Transmission Count')
        plt.title(f'Evolution of Transmissions for Node {node_id}')
        plt.grid(True)
elif to_plot == "histogram":
    # Plot the histogram of transmission counts for each node
    for i, node_id in enumerate(nodes):
        # Extract timestamps and transmission counts for the node
        timestamps = []
        transmissions = []
        config_changes = []
        
        for entry in transmission_timeline[node_id]:
            timestamp, value = entry
            timestamps.append(timestamp)
            if isinstance(value, tuple):
                config_changes.append((timestamp, value))
            else:
                transmissions.append(value)

        # Generate for each configuration the succession of transmission counts
        config_transmissions = defaultdict(list)
        for timestamp, value in transmission_timeline[node_id]:
            if isinstance(value, tuple):
                config = value
            else:
                config_transmissions[config].append(value)
        
        # Create a subplot for the current node
        plt.subplot(num_nodes, 1, i + 1)

        # Plot the histogram of transmission counts
        plt.hist(transmissions, bins=range(1, 9), label=f"Node {node_id}")
        #plt.hist(transmissions, label=f"Node {node_id}")

        # Customize each subplot
        plt.xlabel('Transmission Count')
        plt.xlim(1, 8)
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title(f'Transmission Count Distribution for Node {node_id}')
        plt.legend()

# Adjust layout for better spacing
#plt.xticks(range(1, len(config_transmissions) + 1), [str(cfg) for cfg in config_transmissions.keys()], rotation=90)
plt.tight_layout()
#plt.legend()
#plt.savefig("../figures/mean-var-dev-transmissions-diff.pdf", format="pdf", bbox_inches="tight")
#plt.savefig("../figures/count-distr-diff.pdf", format="pdf", bbox_inches="tight")
plt.show()