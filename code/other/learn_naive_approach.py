import numpy as np # type: ignore
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

with open('../data/diff-config-stats.json', 'r') as file:
    data = json.load(file)

node_model = {}
for node_id in ("m3-123", "m3-133", "m3-143", "m3-150", "m3-153", "m3-159", "m3-163", "m3-166"):
    res = {}

    if node_id not in data:
        continue

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data[node_id]).T
    df.index = df.index.map(eval)  # Convert index from string to tuple

    print(df.head())

    # Separate features (a, b, c, d) and target variables
    X = pd.DataFrame(df.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features
    targets = ['mean', 'std_dev', 'variance']  # Define target variables

    n_splits = 10

    # Perform multiple random splits and evaluate models for each target variable
    for target in targets:
        print(f"Evaluating models for target: {target}")

        # Store results for each model
        results = {"naive": {'mse': [], 'r2': [], 'mae': []}}

        cut_off = int(0.9 * len(X))
        
        #X_train, X_test, y_train, y_test = train_test_split(X, df[target], test_size=0.2, random_state=i)
        X_train = X[:cut_off]
        X_test = X[cut_off:]
        y_train = df[target][:cut_off]
        y_test = df[target][cut_off:]
        # Evaluate the naive model
        y_pred = []

        for index, row in X_test.iterrows():
            y_pred.append(predict(list(zip(X_train.values, y_train.values)), row.values))
        results["naive"]["mse"].append(mean_squared_error(y_test, y_pred))
        results["naive"]["r2"].append(r2_score(y_test, y_pred))
        results["naive"]["mae"].append(mean_absolute_error(y_test, y_pred))

        res[target] = {"model": "naive", "mse": results["naive"]['mse'], "mae": results["naive"]['mae'], "r2": results["naive"]['mae']}
        #print(f"Best model for {target} with the naive approach with MSE: {np.mean(results["naive"]['mse'])} \n")
    node_model[node_id] = res   

print(node_model)


with open('../data/naive-approach-diff-results.json', 'w') as outfile:
    json.dump(node_model, outfile, indent=4)
