import json
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import PCA

sns.set(font_scale=1.2)

# --------------------------
# Load & Merge Multiple Logs
# --------------------------
file_pattern = "../data/IoTJ/expe-iotj-*-transmissions-per-config-stats.json"
file_paths = glob.glob(file_pattern)

print(f"Found {len(file_paths)} log files")

# Accumulate means per link/config across logs
aggregated = defaultdict(lambda: defaultdict(list))

for path in file_paths:
    with open(path, 'r') as f:
        data = json.load(f)
    for link, cfgs in data.items():
        for cfg_str, metrics in cfgs.items():
            aggregated[link][cfg_str].append(metrics["mean"])

# Average across logs
averaged_data = {
    link: {cfg: float(np.mean(vals)) for cfg, vals in cfgs.items()}
    for link, cfgs in aggregated.items()
}

print(averaged_data )

# --------------------------
# Step 1: Find common configurations
# --------------------------
all_links = list(averaged_data.keys())
config_sets = [set(averaged_data[link].keys()) for link in all_links]
common_configs = set.intersection(*config_sets)

print("Common configurations used for clustering:", common_configs)

# --------------------------
# Step 2: Build feature vectors
# --------------------------
link_vectors = []
link_ids = []

for link in all_links:
    vector = [averaged_data[link][cfg] for cfg in sorted(common_configs)]
    link_vectors.append(vector)
    link_ids.append(link)

X = np.array(link_vectors)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Step 3: Clustering
# --------------------------
k = 3  # adjust as needed
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# --------------------------
# Step 4: Show clustering result
# --------------------------
for link, cluster_id in zip(link_ids, clusters):
    print(f"Link {link} is in cluster {cluster_id}")

# --------------------------
# Step 5: Visualization (PCA)
# --------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="Set2", s=100)
for i, txt in enumerate(link_ids):
    plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
plt.title("Link Clustering based on Configuration Performance (mean, averaged across logs)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
