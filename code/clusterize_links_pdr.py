import json
import glob
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

# Step 1: Load all JSON files
file_pattern = "../data/IoTJ/expe-iotj-1ps-*-transmissions-per-config-series.json"
files = glob.glob(file_pattern)

print(f"Found {len(files)} files")

# remove the file with 'extended' in its name
files = [f for f in files if 'extended' not in f]
print(f"Processing {len(files)} files after filtering")

# Step 2: Compute loss rate per link for each file
per_log_stats = []  # list of dicts {link: loss_rate}

for fname in files:
    with open(fname, "r") as f:
        data = json.load(f)

    stats = {}
    for link, configs in data.items():
        all_samples = []
        for pdr_list in configs.values():
            all_samples.extend(pdr_list)
        if len(all_samples) > 0:
            loss_rate = all_samples.count(0.0) / len(all_samples)
            stats[link] = loss_rate
    per_log_stats.append(stats)

print(per_log_stats)
# Step 3: Average loss rates across logs
aggregated_stats = {}
all_links = set().union(*[stats.keys() for stats in per_log_stats])

for link in all_links:
    values = [stats[link] for stats in per_log_stats if link in stats]
    if values:
        aggregated_stats[link] = np.mean(values)

# Step 4: Prepare data for clustering
link_ids = list(aggregated_stats.keys())
loss_rates = np.array([[aggregated_stats[link]] for link in link_ids])

# Step 5: Clustering
k = 4  # adjust depending on your case
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(loss_rates)

# Step 6: Print results
for link, rate, cluster in zip(link_ids, loss_rates, clusters):
    print(f"Link {link} → Avg Loss Rate (across logs) = {rate[0]:.2%} → Cluster {cluster}")

# Print in the format cluster_mean = {"0": ["m3-129","m3-130","m3-131","m3-133","m3-143","m3-153"], "1": ["m3-97"], "2": []}
cluster_dict = {}
for link, cluster in zip(link_ids, clusters):
    cluster_str = str(cluster)
    if cluster_str not in cluster_dict:
        cluster_dict[cluster_str] = []
    cluster_dict[cluster_str].append(link)

print(cluster_dict)

# Step 7: Visualization
plt.figure(figsize=(6, 4))
sns.scatterplot(x=np.arange(len(link_ids)), y=[r[0] for r in loss_rates],
                hue=clusters, palette='Set2', s=100)
plt.xticks(np.arange(len(link_ids)), link_ids, rotation=45)
plt.ylabel("Average Packet Loss Rate (across logs)")
plt.title("Clustering of Links by Packet Loss Rate (Mean Across Logs)")
plt.grid(True)
plt.tight_layout()
plt.show()
