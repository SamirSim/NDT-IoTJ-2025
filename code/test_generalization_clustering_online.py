import json
import re
import numpy as np # type: ignore
import pandas as pd # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # type: ignore

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
with open('../data/IoTJ/'+file_title+'-1ps-extended-transmissions-per-config-stats.json', 'r') as file:
    data_extended = json.load(file)

print(data_extended.keys())

models_merged = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'K-NN': KNeighborsRegressor(n_neighbors=3),
    'Decision Tree': DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

cluster_topology = {'0': ['m3-97', 'm3-98', 'm3-99', 'm3-100', 'm3-109', 'm3-110', 'm3-111', 'm3-112'], '1': ['m3-167', 'm3-168', 'm3-169', 'm3-170'], '2': ['m3-119', 'm3-120', 'm3-121', 'm3-122', 'm3-129', 'm3-130', 'm3-131', 'm3-133'],'3': ['m3-153', 'm3-154', 'm3-156', 'm3-157']}
cut_off = 1

node_cluster_map = {
    101: '0',
    134: '2',
    171: '1'
}

file_title_data = file_title+'-'+'1ps-2'

with open('../data/IoTJ/'+file_title_data+'-transmissions-per-config-stats.json', 'r') as file:
                    data = json.load(file)
filename = '../data/IoTJ/'+file_title_data+'-transmissions-per-config-series.json'

with open(filename, 'r') as file:
    data_series = json.load(file)

cluster_topology_data = {}
res_ = {}
iterative_errors = {node: {name: [] for name in models_merged.keys()} for node in ["m3-101", "m3-134", "m3-171"]}
best_models_per_node = {}

for node_id in ["m3-101", "m3-134", "m3-171"]:
    print(f"\n=== Iterative Generalization for {node_id} ===")

    node_cluster = node_cluster_map[int(node_id.split("-")[1])]

    # Prepare cluster training data (same as before)
    cluster_topology_data = {}
    res_ = {}
    for node, config_dict in data_series.items():
        if node not in cluster_topology[node_cluster]:
            continue
        for config, transmissions in list(config_dict.items())[:cut_off]:
            if config not in cluster_topology_data:
                cluster_topology_data[config] = transmissions
            else:
                cluster_topology_data[config].extend(transmissions)
            res_[config] = compute_statistics(transmissions)

    df_cluster_topology = pd.DataFrame(res_).T
    df_cluster_topology.index = df_cluster_topology.index.map(eval)

    X_train = pd.DataFrame(df_cluster_topology.index.tolist(), columns=['a', 'b', 'c', 'd'])
    y_train = df_cluster_topology[target]

    # Extended (test) data
    df_extended = pd.DataFrame(data_extended[node_id]).T
    df_extended.index = df_extended.index.map(eval)
    X_extended = pd.DataFrame(df_extended.index.tolist(), columns=['a', 'b', 'c', 'd'])
    y_extended = df_extended[target].values

    step_size = 1  # predict every 5 steps

    for name, base_model in models_merged.items():
        model = base_model
        model.fit(X_train, y_train)

        errors = []
        X_curr, y_curr = X_train.copy(), y_train.copy()

        for i in range(0, len(X_extended), step_size):
            # --- Select next chunk ---
            x_chunk = X_extended.iloc[i:i+step_size]
            y_chunk = y_extended[i:i+step_size]

            if len(x_chunk) == 0:
                break

            # --- Make predictions on the chunk ---
            y_pred = model.predict(x_chunk)

            # --- Compute errors for this batch (mean absolute error over chunk) ---
            batch_error = mean_squared_error(y_chunk, y_pred)
            errors.append(batch_error)

            # --- Augment training set with ground truth of the chunk ---
            X_curr = pd.concat([X_curr, x_chunk])
            y_curr = pd.concat([y_curr, pd.Series(y_chunk)])

            # --- Refit model ---
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_curr, y_curr)

        iterative_errors[node_id][name] = errors

        iterative_errors[node_id][name] = errors

    # --- Select best model overall for this node ---
    avg_errors = {name: np.mean(errs) for name, errs in iterative_errors[node_id].items()}
    best_model = min(avg_errors, key=avg_errors.get)
    best_models_per_node[node_id] = (best_model, avg_errors[best_model])

import matplotlib.pyplot as plt

# --- Final summary ---
print("\n=== FINAL SUMMARY ===")
for node_id, (best_model, best_err) in best_models_per_node.items():
    print(f"Node {node_id} → Best Model: {best_model} with Avg Error: {best_err:.4f}")

    # Evolution of errors for the best model
    errors = iterative_errors[node_id][best_model]
    #print(f"  Error evolution for {best_model}: {errors}")

def smooth_curve(values, window=3):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

# --- Plot best model error evolution for all nodes ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (node_id, (best_model, best_err)) in zip(axes, best_models_per_node.items()):
    errors = iterative_errors[node_id][best_model]
    smoothed_errors = smooth_curve(errors, window=3)

    ax.plot(range(1, len(errors) + 1), errors, marker='o', alpha=0.4, label="Raw errors")
    ax.plot(range(1, len(smoothed_errors) + 1), smoothed_errors, color='red', linewidth=2, label="Smoothed")

    ax.set_title(f"{node_id} ({best_model})")
    ax.set_xlabel("Iteration")
    ax.grid(True)

axes[0].set_ylabel("MSE")
axes[-1].legend(loc="upper right")

plt.suptitle("Evolution of Best Model Errors per Node", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# plot a histogram of the three average errors
plt.figure(figsize=(6, 4))
nodes = list(best_models_per_node.keys())
avg_errors = [best_models_per_node[node][1] for node in nodes]
plt.bar(nodes, avg_errors, color=['blue', 'orange', 'green'])
plt.ylabel("Average MSE of Best Model")
plt.title("Average MSE of Best Model per Node")
plt.grid(axis='y')
plt.tight_layout()
plt.show()