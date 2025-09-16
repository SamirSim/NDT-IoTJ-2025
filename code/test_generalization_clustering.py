import json
import re
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.svm import SVR  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # type: ignore


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


file_title = 'expe-iotj'
target = 'mean'

# Load iotj extended data
with open('../data/IoTJ/' + file_title + '-1ps-extended-transmissions-per-config-stats.json', 'r') as file:
    data_extended = json.load(file)

#print(data_extended.keys())

models_merged = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'K-NN': KNeighborsRegressor(n_neighbors=3),
    'Decision Tree': DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

cluster_topology = {
    '0': ['m3-97', 'm3-98', 'm3-99', 'm3-100', 'm3-109', 'm3-110', 'm3-111', 'm3-112'],
    '1': ['m3-167', 'm3-168', 'm3-169', 'm3-170'],
    '2': ['m3-119', 'm3-120', 'm3-121', 'm3-122', 'm3-129', 'm3-130', 'm3-131', 'm3-133'],
    '3': ['m3-153', 'm3-154', 'm3-156', 'm3-157']
}

cut_off = 1

node_cluster_map = {
    101: '0',
    134: '2',
    171: '1'
}

file_title_data = file_title + '-' + '1ps-2'

with open('../data/IoTJ/' + file_title_data + '-transmissions-per-config-stats.json', 'r') as file:
    data = json.load(file)

filename = '../data/IoTJ/' + file_title_data + '-transmissions-per-config-series.json'
with open(filename, 'r') as file:
    data_series = json.load(file)

cluster_topology_data = {}
res_ = {}


# Store best results per node
node_best_results = {}

for node_id in ["m3-101", "m3-134", "m3-171"]:
    results_cluster_topology = {
        name: {'mse': [], 'r2': [], 'mae': [], 'predictions': []}
        for name in models_merged.keys()
    }

    node_cluster = node_cluster_map[int(node_id.split("-")[1])]

    for node, config_dict in data_series.items():
        if node not in cluster_topology[node_cluster]:
            continue

        for config, transmissions in list(config_dict.items())[:cut_off]:
            if config not in cluster_topology_data:
                cluster_topology_data[config] = transmissions
            else:
                for v in transmissions:
                    cluster_topology_data[config].append(v)

            res_[config] = compute_statistics(transmissions)

    data_cluster_topology = res_

    # Convert the dictionary to a DataFrame
    df_cluster_topology = pd.DataFrame(data_cluster_topology).T
    df_cluster_topology.index = df_cluster_topology.index.map(eval)  # Convert index from string to tuple

    # Separate features (a, b, c, d) and target variables
    X_cluster_topology = pd.DataFrame(df_cluster_topology.index.tolist(), columns=['a', 'b', 'c', 'd'])
    X_train_cluster_topology = X_cluster_topology
    y_train_cluster_topology = df_cluster_topology[target]

    # Prepare the test data
    df_extended = pd.DataFrame(data_extended[node_id]).T
    df_extended.index = df_extended.index.map(eval)  # Convert index from string to tuple

    X_extended = pd.DataFrame(df_extended.index.tolist(), columns=['a', 'b', 'c', 'd'])
    y_extended = df_extended[target]

    # Train the models on the cluster topology data
    best_mse_extended = float("inf")
    best_model_extended = None

    for name, model in models_merged.items():
        # Train the model
        model.fit(X_train_cluster_topology, y_train_cluster_topology)

        # Make predictions
        y_pred = model.predict(X_extended)

        # Calculate the evaluation metrics
        mse = mean_squared_error(y_extended, y_pred)
        r2 = r2_score(y_extended, y_pred)
        mae = mean_absolute_error(y_extended, y_pred)

        # Store the metrics for this split
        results_cluster_topology[name]['mse'].append(mse)
        results_cluster_topology[name]['r2'].append(r2)
        results_cluster_topology[name]['mae'].append(mae)
        results_cluster_topology[name]['predictions'].append(y_pred.tolist())

    best_mse_cluster_topology = float("inf")
    best_model_cluster_topology = None

    # Calculate and print the average MSE and R^2 score for each model over all splits
    for name in models_merged.keys():
        avg_mse = np.mean(results_cluster_topology[name]['mse'])
        avg_r2 = np.mean(results_cluster_topology[name]['r2'])
        avg_mae = np.mean(results_cluster_topology[name]['mae'])

        if avg_mse < best_mse_cluster_topology:
            best_mse_cluster_topology = avg_mse
            best_model_cluster_topology = name

    # Predictions of the best model
    best_predictions_cluster_topology = results_cluster_topology[best_model_cluster_topology]['predictions']

   # print("Predictions of the best model (Cluster Topology) for node", node_id, ":", best_model_cluster_topology, best_predictions_cluster_topology)
    print(f"Node {node_id} - Best Model (Cluster Topology): {best_model_cluster_topology} "
          f"with Avg MSE: {best_mse_cluster_topology:.4f}, "
          f"Avg R^2: {avg_r2:.4f}, "
          f"Avg MAE: {avg_mae:.4f}")

    # Store best results per node
    node_best_results[node_id] = {
        "best_model": best_model_cluster_topology,
        "avg_mse": best_mse_cluster_topology,
        "avg_r2": avg_r2,
        "avg_mae": avg_mae
    }

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.8)

# Convert results to DataFrame for easy plotting
df_results = pd.DataFrame(node_best_results).T

# Plot only MSE histogram
plt.figure(figsize=(6, 5))
plt.bar(df_results.index, df_results["avg_mse"], edgecolor="black", color=['skyblue', 'salmon', 'lightgreen'])
plt.title("Generalized approach performance (Cluster Topology)")
plt.ylabel("MSE")
plt.xlabel("Node")
plt.tight_layout()
#plt.show()

plt.savefig("../figures/results-generalization-cluster-topology-exntended.pdf", format="pdf", bbox_inches="tight")