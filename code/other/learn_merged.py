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

with open('../data/diff-config-merged-stats.json', 'r') as file:
    data = json.load(file)

node_model = {}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data).T
df.index = df.index.map(eval)  # Convert index from string to tuple

# Separate features (a, b, c, d) and target variables
X = pd.DataFrame(df.index.tolist(), columns=['a', 'b', 'c', 'd'])  # Tuple as features
targets = ['mean', 'std_dev', 'variance']  # Define target variables

# List of regression models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'K-NN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Number of random splits to perform
n_splits = 10

# Perform multiple random splits and evaluate models for each target variable
for target in targets:
    print(f"Evaluating models for target: {target}")
    
    # Store results for each model
    results = {name: {'mse': [], 'r2': [], 'mae': []} for name in models.keys()}
    
    # Perform multiple random splits
    for i in range(n_splits):
        # Split the data randomly
        X_train, X_test, y_train, y_test = train_test_split(X, df[target], test_size=0.2, random_state=i)
        
        # Train and evaluate each model
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate the evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
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
        #print(f"{name}:")
        #print(f"  Average Mean Squared Error (MSE): {avg_mse}")
        #print(f"  Average Mean Absolute Error (MAE): {avg_mae}")
        #print(f"  Average R^2 Score: {avg_r2}\n")

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_model = name
            
    node_model[target] = {"model": best_model, "mse": results[best_model]['mse'], "mae": results[best_model]['mae'], "r2": results[best_model]['mae']}
    print(f"Best model for {target} is {best_model} with MSE: {best_mse}\n")

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

with open('../data/diff-config-merged-results.json', 'w') as outfile:
    json.dump(node_model, outfile, indent=4)

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