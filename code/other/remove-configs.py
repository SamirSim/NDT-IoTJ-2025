import re

filename = '../data/same-config.txt'

# Read the logs from the file
with open(filename, 'r') as file:
    logs = file.read()

# Regular expressions to capture the necessary fields
config_re = r"m3-(\d+);The configuration has been changed to: CSMA_MIN_BE=(\d+), CSMA_MAX_BE=(\d+), CSMA_MAX_BACKOFF=(\d+)"

# Dictionary to track how many times each node's configuration has been logged
node_count = {}

# List to store the updated lines
updated_logs = []

for line in logs.splitlines():
    config_match = re.search(config_re, line)
    
    if config_match:
        node_id = config_match.group(1)
        # Check if this node has appeared more than 10 times
        if node_id not in node_count:
            node_count[node_id] = 1
            updated_logs.append(line)

        if node_count[node_id] < 10:
            node_count[node_id] += 1

        if node_count[node_id] == 10:
            node_count[node_id] = 0
            updated_logs.append(line)
    else:
        # If the line doesn't match the config change pattern, keep it
        updated_logs.append(line)

# Write the updated logs back to the file or another output
with open('../data/updated-same-config.txt', 'w') as file:
    file.write("\n".join(updated_logs))

print(f"Updated logs have been written to '../data/updated-same-config.txt'.")