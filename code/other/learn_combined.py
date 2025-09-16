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

file_title = 'diff-config-transmissions-per-config'


with open('../data/'+file_title+'-stats.json', 'r') as file:
    data = json.load(file)

node_model = {}
node_model_merged = {}

for node_id in ("123", "133", "143", "150", "153", "159", "163", "166"):
#for node_id in ("95", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110"):
#for node_id in ("1", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"):
    res = {}
    res_merged = {}

    if node_id not in data:
        continue

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data[node_id]).T
    df.index = df.index.map(eval)  # Convert index from string to tuple

    # Separate features (a, b, c, d) and target variables
    X = pd.DataFrame(df.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features
    targets = ['mean']  # Define target variables

    # Number of random splits to perform
    n_splits = [0.6]
    #n_splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    size_testing_ = 0.4

    # Perform multiple random splits and evaluate models for each target variable
    for target in targets:
        res_split = {}
        res_split_merged = {}
        print(f"Evaluating models for target: {target}")

        for cut_off_ in n_splits:
            # List of regression models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'SVR': SVR(),
                'K-NN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }

            models_merged = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'SVR': SVR(),
                'K-NN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }
            
            # Store results for each model
            results = {name: {'mse': [], 'r2': [], 'mae': []} for name in models.keys()}

            results_merged = {name: {'mse': [], 'r2': [], 'mae': []} for name in models.keys()}

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

            print("size of X_train: ", len(X_train))
            print("size of X_test: ", len(X_test))

            filename = '../data/'+file_title+'-series.json'

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

            print("size of X_train_merged: ", len(X_train_merged))
            #sleep(2)
                            
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
                print(name, mse)
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

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_model = name
            
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
                print(name, mse)
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
                print(results_merged[name]['mse'])
                #sleep(1)
                avg_mse = np.mean(results_merged[name]['mse'])
                avg_r2 = np.mean(results_merged[name]['r2'])
                avg_mae = np.mean(results_merged[name]['mae'])

                if avg_mse < best_mse_merged:
                    best_mse_merged = avg_mse
                    best_model_merged = name
            print(results_merged[best_model_merged]['mse'])

            res_split[cut_off_] = {"model": best_model, "mse": results[best_model]['mse'], "mae": results[best_model]['mae'], "r2": results[best_model]['r2']}
            
            print(f"Best model for {target} is {best_model} with MSE: {best_mse}\n")

            res_split_merged[cut_off_] = {"model": best_model_merged, "mse": results_merged[best_model_merged]['mse'], "mae": results_merged[best_model_merged]['mae'], "r2": results_merged[best_model_merged]['r2']}

            print(f"Best model for merged {target} is {best_model_merged} with MSE: {best_mse_merged}\n")
        
        res[target] = res_split
        res_merged[target] = res_split_merged
        
    node_model[node_id] = res
    node_model_merged[node_id] = res_merged

    """
        # Flatten the results for easier plotting
        mse_flat = [(model, mse) for model, mse_list in results.items() for mse in mse_list['mse']]
        mae_flat = [(model, mae) for model, mae_list in results.items() for mae in mae_list['mae']]

        # Convert to DataFrame
        mse_df = pd.DataFrame(mse_flat, columns=['Model', 'MSE'])
        mae_df = pd.DataFrame(mae_flat, columns=['Model', 'MAE'])

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plotting the MSE distribution on the first subplot
        sns.boxplot(x='Model', y='MSE', data=mse_df, ax=axes[0])
        axes[0].set_title('Distribution of MSE Across Models for ' + target + ' for Node ' + node_id)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].grid(True)

        # Plotting the MAE distribution on the second subplot
        sns.boxplot(x='Model', y='MAE', data=mae_df, ax=axes[1])
        axes[1].set_title('Distribution of MAE Across Models for ' + target + ' for Node ' + node_id)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].grid(True)

        # Adjust layout and show plot
        plt.tight_layout() 
        plt.show()
        """

print(node_model)

with open('../data/'+file_title+'-results-split.json', 'w') as outfile:
    json.dump(node_model, outfile, indent=4)

with open('../data/'+file_title+'-merged-results-split.json', 'w') as outfile:
    json.dump(node_model_merged, outfile, indent=4)

"""
for target in targets:
    for node_id, model in node_model.items():
        # Plotting the MSE distribution on the first subplot
        mse_list = results[model]['mse']
        # Flatten the results for easier plotting
        mse_flat = [(model, mse) for model, mse_list in results.items() for mse in mse_list['mse']]
        mae_flat = [(model, mae) for model, mae_list in results.items() for mae in mae_list['mae']]
            
        # Convert to DataFrame
        mse_df = pd.DataFrame(mse_flat, columns=['Model', 'MSE'])
        mae_df = pd.DataFrame(mae_flat, columns=['Model', 'MAE'])

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plotting the MSE distribution on the first subplot
        sns.boxplot(x='Model', y='MSE', data=mse_df, ax=axes[0])
        axes[0].set_title('Distribution of MSE Across Models for ' + target + ' for Node ' + node_id)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].grid(True)

        # Plotting the MAE distribution on the second subplot
        sns.boxplot(x='Model', y='MAE', data=mae_df, ax=axes[1])
        axes[1].set_title('Distribution of MAE Across Models for ' + target + ' for Node ' + node_id)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].grid(True)

        # Adjust layout and show plot
        plt.tight_layout() 
        plt.show()
"""