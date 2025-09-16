import re
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import defaultdict # type: ignore
import time
import json
import numpy as np # type: ignore

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

# Convert tuple keys in final_dict
node_transmissions_str_keys = {node: convert_keys_to_strings(config_dict) for node, config_dict in node_transmission_counts.items()}

# Convert and write JSON object to file
json_file = '../data/' + file_title + '-series.json'
with open(json_file, 'w') as outfile:
    json.dump(node_transmissions_str_keys, outfile)

final_dict = {}
print(node_transmission_counts)
for node, config_dict in node_transmission_counts.items():
    res = {}
    for config, transmissions in config_dict.items():
        res[config] = compute_statistics(transmissions)
    final_dict[node] = res

# Convert tuple keys in final_dict
final_dict_str_keys = {node: convert_keys_to_strings(config_dict) for node, config_dict in final_dict.items()}

# Convert and write JSON object to file
json_file = '../data/' + file_title + '-stats.json'
with open(json_file, 'w') as outfile:
    json.dump(final_dict_str_keys, outfile, indent=4)