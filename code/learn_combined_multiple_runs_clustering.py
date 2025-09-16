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

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore

import random
from time import sleep

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


def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

def predict(data, target, k=3):
    # Calculate distances between the target and each point in the data
    distances = [(euclidean_distance(point[0], target), point[1]) for point in data]
    
    # Sort by distance and select the k closest points
    closest_points = sorted(distances, key=lambda x: x[0])[:k]
    
    # If the distance is zero, return the exact y value
    if closest_points[0][0] == 0:
        return closest_points[0][1]
    
    # Calculate weighted average based on inverse distances
    weights = [(1 / dist, y) for dist, y in closest_points]
    weighted_sum = sum(weight * y for weight, y in weights)
    total_weight = sum(weight for weight, y in weights)
    
    # Predicted y value for the target tuple
    y_pred = weighted_sum / total_weight
    return y_pred

file_title = 'expe-iotj'

node_model = {}
node_model_merged = {}
node_model_naive = {}
node_model_cluster_topology = {}
node_model_cluster_mean = {}
node_model_cluster_ranks = {}

cluster_topology = {'0': ['m3-97', 'm3-98', 'm3-99', 'm3-100', 'm3-109', 'm3-110', 'm3-111', 'm3-112'], '1': ['m3-167', 'm3-168', 'm3-169', 'm3-170'], '2': ['m3-119', 'm3-120', 'm3-121', 'm3-122', 'm3-129', 'm3-130', 'm3-131', 'm3-133'],'3': ['m3-153', 'm3-154', 'm3-156', 'm3-157']}
#cluster_topology = {'0': ['m3-97', 'm3-98', 'm3-99', 'm3-100', 'm3-109', 'm3-110', 'm3-111', 'm3-112'],'1': ['m3-119', 'm3-120', 'm3-121', 'm3-122', 'm3-129', 'm3-130', 'm3-131', 'm3-133'],'2': ['m3-153', 'm3-154', 'm3-156', 'm3-157', 'm3-167', 'm3-168', 'm3-169', 'm3-170']}
cluster_mean = {'0': ['m3-170', 'm3-167'], '2': ['m3-131', 'm3-129', 'm3-168', 'm3-169', 'm3-133', 'm3-153'], '1': ['m3-97', 'm3-111', 'm3-110', 'm3-112', 'm3-100', 'm3-109', 'm3-99', 'm3-98'], '3': ['m3-122', 'm3-154', 'm3-156', 'm3-157', 'm3-130', 'm3-121', 'm3-120', 'm3-119']}
#cluster_ranks = {'2': ['m3-111', 'm3-130', 'm3-120', 'm3-129', 'm3-110', 'm3-154', 'm3-131', 'm3-121', 'm3-133', 'm3-98', 'm3-109', 'm3-157', 'm3-169', 'm3-97', 'm3-112', 'm3-122', 'm3-119', 'm3-153', 'm3-99', 'm3-100'], '0': ['m3-156', 'm3-168'], '1': ['m3-170'], '3': ['m3-167']}
cluster_ranks = {'1': ['m3-111', 'm3-120', 'm3-129', 'm3-110', 'm3-167', 'm3-131', 'm3-121', 'm3-133', 'm3-157', 'm3-122', 'm3-119', 'm3-153', 'm3-99'], '3': ['m3-130', 'm3-156', 'm3-154', 'm3-109', 'm3-97', 'm3-112'], '0': ['m3-170', 'm3-168', 'm3-169'], '2': ['m3-98', 'm3-100']}

for node_id in ("m3-97","m3-98","m3-99","m3-100","m3-109","m3-110","m3-111","m3-112","m3-119","m3-120","m3-121","m3-122","m3-129","m3-130","m3-131","m3-133","m3-143","m3-153","m3-154","m3-156","m3-157","m3-167","m3-168","m3-169","m3-170"):
    targets = ['mean']  # Define target variables

    # Number of random splits to perform
    n_splits = [0.7]
    #n_splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    size_testing_ = 0.3

    res = {}
    res_merged = {}
    res_naive = {}
    res_cluster_topology = {}
    res_cluster_mean = {}
    res_cluster_ranks = {}

    # Perform multiple random splits and evaluate models for each target variable
    for target in targets:
        res_split = {}
        res_split_merged = {}
        res_split_naive = {}
        res_split_cluster_topology = {}
        res_split_cluster_mean = {}
        res_split_cluster_ranks = {}
        print(f"Evaluating models for target: {target}")

        for cut_off_ in n_splits:
            # List of regression models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'SVR': SVR(),
                'K-NN': KNeighborsRegressor(n_neighbors=3),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }

            models_merged = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'SVR': SVR(),
                'K-NN': KNeighborsRegressor(n_neighbors=3),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }
            
            # Store results for each model
            results = {name: {'mse': [], 'r2': [], 'mae': []} for name in models.keys()}

            results_merged = {name: {'mse': [], 'r2': [], 'mae': []} for name in models.keys()}

            results_cluster_topology = {name: {'mse': [], 'r2': [], 'mae': []} for name in models_merged.keys()}

            results_cluster_mean = {name: {'mse': [], 'r2': [], 'mae': []} for name in models_merged.keys()}

            results_cluster_ranks = {name: {'mse': [], 'r2': [], 'mae': []} for name in models_merged.keys()}

            results_naive = {"naive": {'mse': [], 'r2': [], 'mae': []}}

            for i in range(1, 3):
                file_title_data = file_title+'-'+'1ps-'+str(i)

                with open('../data/IoTJ/'+file_title_data+'-transmissions-per-config-stats.json', 'r') as file:
                    data = json.load(file)

                if node_id not in data:
                    continue

                # Convert the dictionary to a DataFrame
                df = pd.DataFrame(data[node_id]).T
                df.index = df.index.map(eval)  # Convert index from string to tuple

                # Separate features (a, b, c, d) and target variables
                X = pd.DataFrame(df.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features

                cut_off = int(cut_off_ * len(X))
                size_testing = int(size_testing_ * len(X))
                
                # Put in X_train the first 80% of the data and in X_test the last 20%
                X_train = X[:cut_off]
                X_test = X[cut_off:cut_off+size_testing]
                y_train = df[target][:cut_off]
                y_test = df[target][cut_off:cut_off+size_testing]

                # Print the first element of x_train and y_train
                #print(X_train.head())
                #print(df[target])

                #print("data: ", y_train)
                #print("test: ", y_test)
                #sleep(2)

                #print("size of X_train: ", len(X_train))
                #print("size of X_test: ", len(X_test))

                filename = '../data/IoTJ/'+file_title_data+'-transmissions-per-config-series.json'

                with open(filename, 'r') as file:
                    data_series = json.load(file)

                merged_data = {}
                res_ = {}
                for node, config_dict in data_series.items():
                    for config, transmissions in list(config_dict.items())[:cut_off]:
                    #for config, transmissions in config_dict.items():
                        #print(config)
                        if config not in merged_data:
                            merged_data[config] = transmissions
                        else:
                            for v in transmissions:
                                merged_data[config].append(v)
                        res_[config] = compute_statistics(transmissions)

                #print(res_)
                #sleep(2)

                data_merged = res_

                # Convert the dictionary to a DataFrame
                df_merged = pd.DataFrame(data_merged).T
                df_merged.index = df_merged.index.map(eval)  # Convert index from string to tuple

                # Separate features (a, b, c, d) and target variables
                X_merged = pd.DataFrame(df_merged.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features

                X_train_merged = X_merged
                y_train_merged = df_merged[target]

                """ Changed in the following """
                # Cluster topology
                # Find the cluster of the node

                node_cluster = None
                for cluster, nodes in cluster_topology.items():
                    if node_id in nodes:
                        node_cluster = cluster
                        break

                cluster_topology_data = {}
                res_ = {}
                for node, config_dict in data_series.items():
                    if node not in cluster_topology[node_cluster]:
                        continue

                    for config, transmissions in list(config_dict.items())[:cut_off]:
                    #for config, transmissions in config_dict.items():
                        #print(config)
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
                X_cluster_topology = pd.DataFrame(df_cluster_topology.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features

                X_train_cluster_topology = X_cluster_topology
                y_train_cluster_topology = df_cluster_topology[target]

                # Cluster mean
                node_cluster = None
                for cluster, nodes in cluster_mean.items():
                    if node_id in nodes:
                        node_cluster = cluster
                        break
                cluster_mean_data = {}
                res_ = {}

                for node, config_dict in data_series.items():
                    if node not in cluster_mean[node_cluster]:
                        continue
                    for config, transmissions in list(config_dict.items())[:cut_off]:
                    #for config, transmissions in config_dict.items():
                        #print(config)
                        if config not in cluster_mean_data:
                            cluster_mean_data[config] = transmissions
                        else:
                            for v in transmissions:
                                cluster_mean_data[config].append(v)
                        res_[config] = compute_statistics(transmissions)

                data_cluster_mean = res_
                # Convert the dictionary to a DataFrame
                df_cluster_mean = pd.DataFrame(data_cluster_mean).T
                df_cluster_mean.index = df_cluster_mean.index.map(eval)  # Convert index from string
                # Separate features (a, b, c, d) and target variables
                X_cluster_mean = pd.DataFrame(df_cluster_mean.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features
                X_train_cluster_mean = X_cluster_mean
                y_train_cluster_mean = df_cluster_mean[target]

                # Cluster ranks
                node_cluster = None
                for cluster, nodes in cluster_ranks.items():
                    if node_id in nodes:
                        node_cluster = cluster
                        break
                cluster_ranks_data = {}
                res_ = {}

                for node, config_dict in data_series.items():
                    if node not in cluster_ranks[node_cluster]:
                        continue
                    print(node_id, node, node_cluster)
                    #sleep(2)
                    for config, transmissions in list(config_dict.items())[:cut_off]:
                    #for config, transmissions in config_dict.items():
                        #print(config)
                        if config not in cluster_ranks_data:
                            cluster_ranks_data[config] = transmissions
                        else:
                            for v in transmissions:
                                cluster_ranks_data[config].append(v)
                        res_[config] = compute_statistics(transmissions)

                data_cluster_ranks = res_
                # Convert the dictionary to a DataFrame
                df_cluster_ranks = pd.DataFrame(data_cluster_ranks).T
                df_cluster_ranks.index = df_cluster_ranks.index.map(eval)  # Convert index
                # Separate features (a, b, c, d) and target variables
                X_cluster_ranks = pd.DataFrame(df_cluster_ranks.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features
                X_train_cluster_ranks = X_cluster_ranks
                y_train_cluster_ranks = df_cluster_ranks[target]

                print(X_cluster_ranks)
                """ Changed in the previous """
                
                model = None
                # Train and evaluate each model
                for name, model in models.items():
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate the evaluation metrics
                    mse = mean_squared_error(y_test, y_pred)
                    #print(y_pred)
                    #print(y_test.values)
                    #print(name, mse)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store the metrics for this split
                    results[name]['mse'].append(mse)
                    results[name]['r2'].append(r2)
                    results[name]['mae'].append(mae)

                best_mse = float("inf")
                best_model = None

                # Calculate and print the average MSE and R^2 score for each model over all splits
                for name in models.keys():
                    avg_mse = np.mean(results[name]['mse'])
                    avg_r2 = np.mean(results[name]['r2'])
                    avg_mae = np.mean(results[name]['mae'])

                    print(f"Model: {name}, MSE: {avg_mse}, R^2: {avg_r2}, MAE: {avg_mae} for split {i}")

                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_model = name
                
                # Evaluate the naive model
                y_pred = []
                for index, row in X_test.iterrows():
                    y_pred.append(predict(list(zip(X_train.values, y_train.values)), row.values))

                results_naive["naive"]["mse"].append(mean_squared_error(y_test, y_pred))
                results_naive["naive"]["r2"].append(r2_score(y_test, y_pred))
                results_naive["naive"]["mae"].append(mean_absolute_error(y_test, y_pred))

                model = None
                for name, model in models_merged.items():
                    # Train the model
                    model.fit(X_train_merged, y_train_merged)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate the evaluation metrics
                    mse = mean_squared_error(y_test, y_pred)
                    #print(y_pred)
                    #print(y_test.values)
                    #print(name, mse)
                    #sleep(1)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store the metrics for this split
                    results_merged[name]['mse'].append(mse)
                    results_merged[name]['r2'].append(r2)
                    results_merged[name]['mae'].append(mae)
                
                best_mse_merged = float("inf")
                best_model_merged = None

                # Calculate and print the average MSE and R^2 score for each model over all splits
                for name in models_merged.keys():
                    #print(results_merged[name]['mse'])
                    #sleep(1)
                    avg_mse = np.mean(results_merged[name]['mse'])
                    avg_r2 = np.mean(results_merged[name]['r2'])
                    avg_mae = np.mean(results_merged[name]['mae'])

                    if avg_mse < best_mse_merged:
                        best_mse_merged = avg_mse
                        best_model_merged = name
                #print(results_merged[best_model_merged]['mse'])

                # Compute results with clustering approaches
                # Cluster topology

                model = None
                for name, model in models_merged.items():
                    # Train the model
                    model.fit(X_train_cluster_topology, y_train_cluster_topology)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate the evaluation metrics
                    mse = mean_squared_error(y_test, y_pred)
                    #print(y_pred)
                    #print(y_test.values)
                    #print(name, mse)
                    #sleep(1)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store the metrics for this split
                    results_cluster_topology[name]['mse'].append(mse)
                    results_cluster_topology[name]['r2'].append(r2)
                    results_cluster_topology[name]['mae'].append(mae)

                best_mse_cluster_topology = float("inf")
                best_model_cluster_topology = None

                # Calculate and print the average MSE and R^2 score for each model over all splits
                for name in models_merged.keys():
                    #print(results_merged[name]['mse'])
                    #sleep(1)
                    avg_mse = np.mean(results_cluster_topology[name]['mse'])
                    avg_r2 = np.mean(results_cluster_topology[name]['r2'])
                    avg_mae = np.mean(results_cluster_topology[name]['mae'])

                    if avg_mse < best_mse_cluster_topology:
                        best_mse_cluster_topology = avg_mse
                        best_model_cluster_topology = name

                # Cluster mean
                model = None
                for name, model in models_merged.items():
                    # Train the model
                    model.fit(X_train_cluster_mean, y_train_cluster_mean)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate the evaluation metrics
                    mse = mean_squared_error(y_test, y_pred)
                    #print(y_pred)
                    #print(y_test.values)
                    #print(name, mse)
                    #sleep(1)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store the metrics for this split
                    results_cluster_mean[name]['mse'].append(mse)
                    results_cluster_mean[name]['r2'].append(r2)
                    results_cluster_mean[name]['mae'].append(mae)

                best_mse_cluster_mean = float("inf")
                best_model_cluster_mean = None

                # Calculate and print the average MSE and R^2 score for each model over all splits
                for name in models_merged.keys():
                    #print(results_merged[name]['mse'])
                    #sleep(1)
                    avg_mse = np.mean(results_cluster_mean[name]['mse'])
                    avg_r2 = np.mean(results_cluster_mean[name]['r2'])
                    avg_mae = np.mean(results_cluster_mean[name]['mae'])

                    if avg_mse < best_mse_cluster_mean:
                        best_mse_cluster_mean = avg_mse
                        best_model_cluster_mean = name

                # Cluster ranks
                model = None

                for name, model in models_merged.items():
                    # Train the model
                    model.fit(X_train_cluster_ranks, y_train_cluster_ranks)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate the evaluation metrics
                    mse = mean_squared_error(y_test, y_pred)
                    #print(y_pred)
                    #print(y_test.values)
                    #print(name, mse)
                    #sleep(1)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store the metrics for this split
                    results_cluster_ranks[name]['mse'].append(mse)
                    results_cluster_ranks[name]['r2'].append(r2)
                    results_cluster_ranks[name]['mae'].append(mae)

                best_mse_cluster_ranks = float("inf")
                best_model_cluster_ranks = None

                # Calculate and print the average MSE and R^2 score for each model over all splits
                for name in models_merged.keys():
                    #print(results_merged[name]['mse'])
                    #sleep(1)
                    avg_mse = np.mean(results_cluster_ranks[name]['mse'])
                    avg_r2 = np.mean(results_cluster_ranks[name]['r2'])
                    avg_mae = np.mean(results_cluster_ranks[name]['mae'])

                    if avg_mse < best_mse_cluster_ranks:
                        best_mse_cluster_ranks = avg_mse
                        best_model_cluster_ranks = name


            res_split[cut_off_] = {"model": best_model, "mse": results[best_model]['mse'], "mae": results[best_model]['mae'], "r2": results[best_model]['r2']}
            
            print(f"Best model for {target} is {best_model} with MSE: {best_mse}\n")

            res_split_merged[cut_off_] = {"model": best_model_merged, "mse": results_merged[best_model_merged]['mse'], "mae": results_merged[best_model_merged]['mae'], "r2": results_merged[best_model_merged]['r2']}

            print(f"Best model for merged {target} is {best_model_merged} with MSE: {best_mse_merged}\n")
    
            res_split_naive[cut_off_] = {"model": "naive", "mse": results_naive["naive"]['mse'], "mae": results_naive["naive"]['mae'], "r2": results_naive["naive"]['mae']}

            print(f"Naive model for {target} has MSE: {np.mean(results_naive['naive']['mse'])}\n")

            res_split_cluster_topology[cut_off_] = {"model": best_model_cluster_topology, "mse": results_cluster_topology[best_model_cluster_topology]['mse'], "mae": results_cluster_topology[best_model_cluster_topology]['mae'], "r2": results_cluster_topology[best_model_cluster_topology]['r2']}

            res_split_cluster_mean[cut_off_] = {"model": best_model_cluster_mean, "mse": results_cluster_mean[best_model_cluster_mean]['mse'], "mae": results_cluster_mean[best_model_cluster_mean]['mae'], "r2": results_cluster_mean[best_model_cluster_mean]['r2']}

            res_split_cluster_ranks[cut_off_] = {"model": best_model_cluster_ranks, "mse": results_cluster_ranks[best_model_cluster_ranks]['mse'], "mae": results_cluster_ranks[best_model_cluster_ranks]['mae'], "r2": results_cluster_ranks[best_model_cluster_ranks]['r2']}

            print(f"Best model for cluster topology {target} is {best_model_cluster_topology} with MSE: {best_mse_cluster_topology}\n")
            print(f"Best model for cluster mean {target} is {best_model_cluster_mean} with MSE: {best_mse_cluster_mean}\n")
            print(f"Best model for cluster ranks {target} is {best_model_cluster_ranks} with MSE: {best_mse_cluster_ranks}\n")


        res[target] = res_split
        res_merged[target] = res_split_merged
        res_naive[target] = res_split_naive
        res_cluster_topology[target] = res_split_cluster_topology
        res_cluster_mean[target] = res_split_cluster_mean
        res_cluster_ranks[target] = res_split_cluster_ranks

    node_model[node_id] = res
    node_model_merged[node_id] = res_merged
    node_model_naive[node_id] = res_naive
    node_model_cluster_topology[node_id] = res_cluster_topology
    node_model_cluster_mean[node_id] = res_cluster_mean
    node_model_cluster_ranks[node_id] = res_cluster_ranks

print(node_model)
#sleep(10)

with open('../data/IoTJ/'+file_title+'-ps-results-mul-runs-split.json', 'w') as outfile:
    json.dump(node_model, outfile, indent=4)

with open('../data/IoTJ/'+file_title+'-ps-merged-results-mul-runs-split.json', 'w') as outfile:
    json.dump(node_model_merged, outfile, indent=4)

with open('../data/IoTJ/'+file_title+'-ps-naive-results-mul-runs-split.json', 'w') as outfile:
    json.dump(node_model_naive, outfile, indent=4)

with open('../data/IoTJ/'+file_title+'-ps-cluster-topology-results-mul-runs-split.json', 'w') as outfile:
    json.dump(node_model_cluster_topology, outfile, indent=4)

with open('../data/IoTJ/'+file_title+'-ps-cluster-mean-results-mul-runs-split.json', 'w') as outfile:
    json.dump(node_model_cluster_mean, outfile, indent=4)

with open('../data/IoTJ/'+file_title+'-ps-cluster-ranks-results-mul-runs-split.json', 'w') as outfile:
    json.dump(node_model_cluster_ranks, outfile, indent=4)

# calculate the average mse of the cluster ranks approach for all nodes
avg_mse_cluster_ranks = {}
for node_id, res in node_model_cluster_ranks.items():
    for target, res_target in res.items():
        for cut_off, res_cut_off in res_target.items():
            if cut_off not in avg_mse_cluster_ranks:
                avg_mse_cluster_ranks[cut_off] = []
            avg_mse_cluster_ranks[cut_off].extend(res_cut_off['mse'])

for cut_off, mses in avg_mse_cluster_ranks.items():
    avg_mse = np.mean(mses)
    print(f"Average MSE for cluster ranks approach with cut-off {cut_off}: {avg_mse}")

# calculate the average mse of the cluster mean approach for all nodes
avg_mse_cluster_mean = {}
for node_id, res in node_model_cluster_mean.items():
    for target, res_target in res.items():
        for cut_off, res_cut_off in res_target.items():
            if cut_off not in avg_mse_cluster_mean:
                avg_mse_cluster_mean[cut_off] = []
            avg_mse_cluster_mean[cut_off].extend(res_cut_off['mse'])
for cut_off, mses in avg_mse_cluster_mean.items():
    avg_mse = np.mean(mses)
    print(f"Average MSE for cluster mean approach with cut-off {cut_off}: {avg_mse}")

# calculate the average mse of the cluster topology approach for all nodes
avg_mse_cluster_topology = {}
for node_id, res in node_model_cluster_topology.items():
    for target, res_target in res.items():
        for cut_off, res_cut_off in res_target.items():
            if cut_off not in avg_mse_cluster_topology:
                avg_mse_cluster_topology[cut_off] = []
            avg_mse_cluster_topology[cut_off].extend(res_cut_off['mse'])
for cut_off, mses in avg_mse_cluster_topology.items():
    avg_mse = np.mean(mses)
    print(f"Average MSE for cluster topology approach with cut-off {cut_off}: {avg_mse}")

# calculate the average mse of the merged approach for all nodes
avg_mse_merged = {}
for node_id, res in node_model_merged.items(): 
    for target, res_target in res.items():
        for cut_off, res_cut_off in res_target.items():
            if cut_off not in avg_mse_merged:
                avg_mse_merged[cut_off] = []
            avg_mse_merged[cut_off].extend(res_cut_off['mse'])

for cut_off, mses in avg_mse_merged.items():
    avg_mse = np.mean(mses)
    print(f"Average MSE for merged approach with cut-off {cut_off}: {avg_mse}")

# calculate the average mse of the standard approach for all nodes
avg_mse = {}
for node_id, res in node_model.items():
    for target, res_target in res.items():
        for cut_off, res_cut_off in res_target.items():
            if cut_off not in avg_mse:
                avg_mse[cut_off] = []
            avg_mse[cut_off].extend(res_cut_off['mse'])
for cut_off, mses in avg_mse.items():
    avg_mse_value = np.mean(mses)
    print(f"Average MSE for standard approach with cut-off {cut_off}: {avg_mse_value}")

