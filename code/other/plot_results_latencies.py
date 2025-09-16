import json
import re
import numpy as np # type: ignore
import pandas as pd # type: ignore

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

with open('../data/same-config-latency-results.json', 'r') as file:
    data = json.load(file)

with open('../data/naive-approach-latency-results.json', 'r') as file:
    naive_data = json.load(file)

with open('../data/same-config-latency-merged-results.json', 'r') as file:
    merged_data = json.load(file)

# Prepare data for plotting
mse_data = []
merged_mse_data = []
naive_mse_data = []

node_ids = ["95", "101", "102", "103", "104", "105", "106", "108", "109"]

# Loop through each node_id and feature
for node_id, features in data.items():
    for feature, values in features.items():
        mse_list = values['mse']
        for mse in mse_list:
            mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

# Loop through each node_id and feature
for node_id, features in naive_data.items():
    for feature, values in features.items():
        mse_list = values['mse']
        for mse in mse_list:
            naive_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

# Loop through each node_id and feature
for feature, values in merged_data.items():
    mse_list = values['mse']
    for mse in mse_list:
        for node_id in node_ids:
            merged_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})


# Convert to DataFrame
mse_df = pd.DataFrame(mse_data)
merged_mse_df = pd.DataFrame(merged_mse_data)
naive_mse_df = pd.DataFrame(naive_mse_data)

# Add a column to distinguish between the two datasets
mse_df['Model'] = 'Original'
merged_mse_df['Model'] = 'Merged'
naive_mse_df['Model'] = 'Naive'

# Combine the two DataFrames
combined_df = pd.concat([merged_mse_df, mse_df, naive_mse_df])

# Create subplots for each feature
features = ['mean', 'variance', 'std_dev']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, feature in enumerate(features):
    # Create boxplots with hue based on the 'Source' column
    sns.boxplot(x='Node ID', y='MSE', hue='Model', data=combined_df[combined_df['Feature'] == feature], ax=axes[i])
    axes[i].set_title(f'Distribution of MSE for {feature.capitalize()}')
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig("../figures/results-models.pdf", format="pdf", bbox_inches="tight")
plt.show()