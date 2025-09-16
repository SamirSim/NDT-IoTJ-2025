import json
import numpy as np # type: ignore
from time import sleep

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

file_title = 'diff-config'
filename = '../data/'+file_title+'-series.json'

with open(filename, 'r') as file:
    data = json.load(file)


merged_data = {}
for node, config_dict in data.items():
    print(len(config_dict))
    cutoff_index = int(0.8 * len(config_dict))
    print(len(list(config_dict.items())[:cutoff_index]))
    sleep(2)
    for config, transmissions in list(config_dict.items())[:cutoff_index]:
        if config not in merged_data:
            merged_data[config] = transmissions
        else:
            for v in transmissions:
                merged_data[config].append(v)
        # res[config] = compute_statistics(transmissions)
    # final_dict[node] = res
print(merged_data)

with open('../data/'+file_title+'-merged-series.json', 'w') as outfile:
    json.dump(merged_data, outfile)

final_dict = {}
res = {}
for config, transmissions in merged_data.items():
    res[config] = compute_statistics(transmissions)

# Convert tuple keys in final_dict
final_dict_str_keys = {node: convert_keys_to_strings(config_dict) for node, config_dict in res.items()}

# Convert and write JSON object to file
json_file = '../data/' + file_title + '-merged-stats.json'
with open(json_file, 'w') as outfile:
    json.dump(final_dict_str_keys, outfile, indent=4)

