import json
import glob
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

# Step 1: Load all JSON files
file_pattern = "../data/IoTJ/expe-iotj-*-transmissions-per-config-stats.json"
files = glob.glob(file_pattern)

print(f"Found {len(files)} files")

# Step 2: Compute average mean per link for each file
per_log_stats = []  # list of dicts {link: avg_mean}

for fname in files:
    with open(fname, "r") as f:
        data = json.load(f)

    stats = {}
    for link, configs in data.items():
        means = [v["mean"] for v in configs.values()]
        if means:
            stats[link] = np.mean(means)
    per_log_stats.append(stats)

# Step 3: Average across logs for each link
aggregated_stats = {}
all_links = set().union(*[stats.keys() for stats in per_log_stats])

for link in all_links:
    values = [stats[link] for stats in per_log_stats if link in stats]
    if values:
        aggregated_stats[link] = np.mean(values)

# Step 4: Prepare data for clustering
link_ids = list(aggregated_stats.keys())
avg_means = np.array([[aggregated_stats[link]] for link in link_ids])

# Step 5: Clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(avg_means)

# Step 6: Display results
for link, avg, cluster in zip(link_ids, avg_means, clusters):
    print(f"Link {link} → Avg Mean (across logs) = {avg[0]:.4f} → Cluster {cluster}")

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
sns.scatterplot(x=np.arange(len(link_ids)), y=[a[0] for a in avg_means],
                hue=clusters, palette='Set2', s=100)
plt.xticks(np.arange(len(link_ids)), link_ids, rotation=45)
plt.ylabel("Average Mean Performance (across logs)")
plt.title("Link Clustering Based on Averaged Mean Across Logs & Configurations")
plt.grid(True)
plt.tight_layout()
plt.show()
