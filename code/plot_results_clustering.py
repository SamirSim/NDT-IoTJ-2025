import json
import re
import numpy as np # type: ignore
import pandas as pd # type: ignore

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

sns.set(font_scale=1.2)

#plt.rcParams['font.family'] = 'Helvetica'

file_title = 'expe-iotj'

cluster_ranks = {'0': ['m3-97', 'm3-98', 'm3-99', 'm3-100', 'm3-109', 'm3-110', 'm3-111', 'm3-112'], '1': ['m3-167', 'm3-168', 'm3-169', 'm3-170'], '2': ['m3-119', 'm3-120', 'm3-121', 'm3-122', 'm3-129', 'm3-130', 'm3-131', 'm3-133'],'3': ['m3-153', 'm3-154', 'm3-156', 'm3-157']}

# Take three nodes per cluster for plotting
selected_nodes = []
for cluster, nodes in cluster_ranks.items():
    # take randomly 3 nodes from each cluster
    randomly_nodes = np.random.choice(nodes, size=min(2, len(nodes)), replace=False)
    selected_nodes.extend(randomly_nodes)

selected_nodes = ["m3-98", "m3-110", "m3-119", "m3-131", "m3-156", "m3-157", "m3-168", "m3-170"]

#selected_nodes = ["m3-97","m3-98","m3-99","m3-100","m3-109","m3-110","m3-111","m3-112","m3-119","m3-120","m3-121","m3-122","m3-129","m3-130","m3-131","m3-133","m3-143","m3-153","m3-154","m3-156","m3-157","m3-167","m3-168","m3-169","m3-170"]

# Load data from JSON files
with open('../data/IoTJ/'+file_title+'-ps-results-mul-runs-split.json', 'r') as file:
    data = json.load(file)

with open('../data/IoTJ/'+file_title+'-ps-naive-results-mul-runs-split.json', 'r') as file:
    naive_data = json.load(file)

with open('../data/IoTJ/'+file_title+'-ps-merged-results-mul-runs-split.json', 'r') as file:
    merged_data = json.load(file)

with open('../data/IoTJ/'+file_title+'-ps-cluster-topology-results-mul-runs-split.json', 'r') as file:
    cluster_topology_data = json.load(file)

with open('../data/IoTJ/'+file_title+'-ps-cluster-mean-results-mul-runs-split.json', 'r') as file:
    cluster_mean_data = json.load(file)

with open('../data/IoTJ/'+file_title+'-ps-cluster-ranks-results-mul-runs-split.json', 'r') as file:
    cluster_ranks_data = json.load(file)

# Prepare data for plotting
mse_data = []
merged_mse_data = []
naive_mse_data = []
cluster_topology_mse_data = []
cluster_mean_mse_data = []
cluster_ranks_mse_data = []

#node_ids = ["95", "101", "102", "103", "104", "105", "106", "108", "109", "110"]
#node_ids = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
node_ids = ["m3-123", "m3-133", "m3-143", "m3-150", "m3-153", "m3-159", "m3-163", "m3-166"]
node_ids = ["m3-97","m3-98","m3-99","m3-100","m3-109","m3-110","m3-111","m3-112","m3-119","m3-120","m3-121","m3-122","m3-129","m3-130","m3-131","m3-133","m3-143","m3-153","m3-154","m3-156","m3-157","m3-167","m3-168","m3-169","m3-170"]

# Loop through each node_id and feature
for node_id, features in data.items():
    if node_id not in selected_nodes:
        continue
    for feature, values in features.items():
        mse_list = values['0.7']['mse']
        #mse_list = values['mse']
        for mse in mse_list:
            mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})


# Loop through each node_id and feature
for node_id, features in naive_data.items():
    if node_id not in selected_nodes:
        continue
    for feature, values in features.items():
        mse_list = values['0.7']['mse']
        for mse in mse_list:
            naive_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

# Loop through each node_id and feature
for node_id, features in merged_data.items():
    if node_id not in selected_nodes:
        continue
    for feature, values in features.items():
        mse_list = values['0.7']['mse']
        #mse_list = values['mse']
        for mse in mse_list:
            merged_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

# Loop through each node_id and feature
for node_id, features in cluster_topology_data.items():
    if node_id not in selected_nodes:
        continue
    for feature, values in features.items():
        mse_list = values['0.7']['mse']
        #mse_list = values['mse']
        for mse in mse_list:
            cluster_topology_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

# Loop through each node_id and feature
for node_id, features in cluster_mean_data.items():
    if node_id not in selected_nodes:
        continue
    for feature, values in features.items():
        mse_list = values['0.7']['mse']
        #mse_list = values['mse']
        for mse in mse_list:
            cluster_mean_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

# Loop through each node_id and feature
for node_id, features in cluster_ranks_data.items():
    if node_id not in selected_nodes:
        continue
    for feature, values in features.items():
        mse_list = values['0.7']['mse']
        #mse_list = values['mse']
        for mse in mse_list:
            cluster_ranks_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})

"""
print(merged_data.items())
# Loop through each node_id and feature
for feature, values in merged_data.items():
    mse_list = values['mse']
    for mse in mse_list:
        for node_id in node_ids:
            merged_mse_data.append({'Node ID': node_id, 'Feature': feature, 'MSE': mse})
"""

print(merged_mse_data)

# Convert to DataFrame
mse_df = pd.DataFrame(mse_data)
merged_mse_df = pd.DataFrame(merged_mse_data)
naive_mse_df = pd.DataFrame(naive_mse_data)
cluster_topology_mse_df = pd.DataFrame(cluster_topology_mse_data)
cluster_mean_mse_df = pd.DataFrame(cluster_mean_mse_data)
cluster_ranks_mse_df = pd.DataFrame(cluster_ranks_mse_data)

# Add a column to distinguish between the two datasets
mse_df['Approach'] = 'Single-link Regression'
merged_mse_df['Approach'] = 'Global Regression'
naive_mse_df['Approach'] = 'Single-link k-NN'
cluster_topology_mse_df['Approach'] = 'Cluster Topology'
cluster_mean_mse_df['Approach'] = 'Cluster Mean'
cluster_ranks_mse_df['Approach'] = 'Cluster Ranks'

# Combine the two DataFrames
#combined_df = pd.concat([mse_df, merged_mse_df, naive_mse_df, cluster_topology_mse_df, cluster_mean_mse_df, cluster_ranks_mse_df])
#combined_df = pd.concat([mse_df, merged_mse_df, naive_mse_df])
combined_df = pd.concat([mse_df, cluster_topology_mse_df, cluster_mean_mse_df, cluster_ranks_mse_df])

feature = 'mean'

# Create the plot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Node ID',
    y='MSE',
    hue='Approach',
    data=combined_df[combined_df['Feature'] == feature],
)
#plt.title(f'Distribution of MSE for {feature.capitalize()} reception rate')
plt.xlabel("Link")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()

# Adjust layout
plt.tight_layout()
plt.savefig("../figures/results-models-cluster-comparison-uc-2.pdf", format="pdf", bbox_inches="tight")
plt.show()