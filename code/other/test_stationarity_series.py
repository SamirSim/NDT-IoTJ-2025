import json
import numpy as np # type: ignore
import pandas as pd # type: ignore
from statsmodels.tsa.stattools import adfuller # type: ignore
import matplotlib.pyplot as plt # type: ignore

file_name = 'diff-config-long-1-transmissions-per-config-series.json'
#file_name = 'same-config-stats.json'

with open (f'../data/{file_name}', 'r') as file:
    data = json.load(file)


# Modify the check_stationarity function to merge transmission series across configurations per node
def check_stationarity(data):
    results = {}
    for node, links in data.items():
        all_transmissions = []  # List to accumulate transmissions across all configurations for a node
        for link, transmissions in links.items():
            if len(transmissions) >= 90:
                all_transmissions.extend(transmissions)  # Merge the transmissions for the node
        print(node, len(all_transmissions))
        p_value = adfuller(all_transmissions)  # ADF test on the merged series
        is_stationary = p_value[1] < 0.05  # If p-value is less than 0.05, the series is stationary
        results[node] = {"p-value": p_value[1], "stationary": is_stationary}
    
    return results

# Running the check
stationarity_results = check_stationarity(data)

# Print results
for node, result in stationarity_results.items():
    print(f"Node: {node} -> p-value: {result['p-value']}, Stationary: {result['stationary']}")