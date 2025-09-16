import json
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from collections import defaultdict

# --------------------------
# Load & Merge Multiple Logs
# --------------------------
file_pattern = "../data/IoTJ/expe-iotj-1ps-*-transmissions-per-config-stats.json"
file_paths = glob.glob(file_pattern)

bounds = np.array([
    [0, 7],    # range of param a
    [3, 8],   # range of param b
    [0, 5], # range of param c
    [0, 7]     # range of param d
])

def normalized_distance(c, u, bounds):
    c = np.array(c)
    u = np.array(u)
    norm_c = (c - bounds[:,0]) / (bounds[:,1] - bounds[:,0])
    norm_u = (u - bounds[:,0]) / (bounds[:,1] - bounds[:,0])
    return np.linalg.norm(norm_c - norm_u)

print(f"Found {len(file_paths)} log files")

# remove the file with 'extended' in its name
file_paths = [f for f in file_paths if 'extended' not in f]
print(f"Processing {len(file_paths)} files after filtering")

# Accumulate stats per node & config across files
aggregated = defaultdict(lambda: defaultdict(list))

for path in file_paths:
    with open(path, 'r') as f:
        data = json.load(f)
    for node, cfgs in data.items():
        for cfg_str, metrics in cfgs.items():
            aggregated[node][cfg_str].append(metrics["mean"])  # store mean PDR

# Compute average across logs
averaged_data = {
    node: {cfg: float(np.mean(vals)) for cfg, vals in cfgs.items()}
    for node, cfgs in aggregated.items()
}

# --------------------------
# Helper: parse config string
# --------------------------
def parse_cfg(cfg_str):
    return tuple(map(int, cfg_str.strip("() ").split(",")))

# --------------------------
# Step 1: Align configs
# --------------------------
def align_configs(data, tolerance):
    # Collect all configs as tuples
    all_cfgs = [parse_cfg(cfg) for node in data.values() for cfg in node]

    # Deduplicate with tolerance
    unique_cfgs = []
    for c in all_cfgs:
        if not any(normalized_distance(c, u, bounds) <= tolerance for u in unique_cfgs):
            unique_cfgs.append(c)

    # Create aligned dataframe (rows = nodes, cols = representative configs)
    df = pd.DataFrame(index=data.keys(), columns=[str(c) for c in unique_cfgs], dtype=float)

    for node, cfgs in data.items():
        for cfg_str, mean_val in cfgs.items():
            cfg = parse_cfg(cfg_str)
            # find closest representative config
            closest = min(unique_cfgs, key=lambda u: normalized_distance(cfg, u, bounds))
            if normalized_distance(cfg, closest, bounds) <= tolerance:
                df.loc[node, str(closest)] = mean_val

    print("unique configs after alignment:", unique_cfgs, len(unique_cfgs))

    return df

# Align with tolerance
df = align_configs(averaged_data, tolerance=0.2)

# Fill missing with column means
df = df.fillna(df.mean())

# --------------------------
# Step 2: Ranking conversion
# --------------------------
rank_df = df.rank(axis=1, ascending=False, method="dense")

# --------------------------
# Step 3: Pairwise distances
# --------------------------
def spearman_distance(u, v):
    corr = pd.Series(u).corr(pd.Series(v), method="spearman")
    return 1 - corr if corr is not None else 1

rank_matrix = rank_df.to_numpy()
distances = pdist(rank_matrix, metric=spearman_distance)

print(distances)

# --------------------------
# Step 4: Hierarchical clustering
# --------------------------
linkage_matrix = linkage(distances, method="complete")

plt.figure(figsize=(8, 4))
dendrogram(linkage_matrix, labels=rank_df.index.tolist(), leaf_rotation=45)
plt.title("Clustering of Nodes by Ranking Similarity (Averaged Across Logs)")
plt.tight_layout()
#plt.show()

# --------------------------
# Step 5: Assign clusters
# --------------------------
clusters = fcluster(linkage_matrix, t=4, criterion="maxclust")
node_clusters = {node: cluster for node, cluster in zip(rank_df.index, clusters)}
print("Node clusters:", node_clusters)

# # Print in the format cluster_mean = {"0": ["m3-129","m3-130","m3-131","m3-133","m3-143","m3-153"], "1": ["m3-97"], "2": []}
cluster_dict = {}
for node, cluster in node_clusters.items():
    cluster_str = str(cluster-1)  # zero-indexed
    if cluster_str not in cluster_dict:
        cluster_dict[cluster_str] = []
    cluster_dict[cluster_str].append(node)

print(cluster_dict)
