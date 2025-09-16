import json
from statsmodels.tsa.stattools import adfuller # type: ignore
import matplotlib.pyplot as plt # type: ignore

file_name = 'diff-config-long-1-transmissions-per-config-stats.json'
#file_name = 'same-config-stats.json'

with open (f'../data/{file_name}', 'r') as file:
    data = json.load(file)

# Function to test stationarity
def test_stationarity(data):
    results = {}
    for node, links in data.items():
        means = [config["mean"] for config in links.values()]
        if len(means) > 1:  # ADF test requires at least 2 values
            adf_test = adfuller(means)
            results[node] = {
                "p_value": adf_test[1],
                "is_stationary": adf_test[1] < 0.05  # Stationary if p-value < 0.05
            }
        else:
            results[node] = "Insufficient data for stationarity test"
    return results

# Run the test
stationarity_results = test_stationarity(data)

# Display results
print(stationarity_results)

# Function to test stationarity over a sliding window
def test_stationarity_with_window(data):
    results = {}
    for node, links in data.items():
        means = [config["mean"] for config in links.values()]
        non_stationary_window = None
        for window_size in range(10, len(means) + 1):
            adf_test = adfuller(means[:window_size])
            if adf_test[1] >= 0.05:  # Non-stationary if p-value >= 0.05
                non_stationary_window = window_size
                break
        results[node] = {
            "non_stationary_window": non_stationary_window,
            "means_considered": means
        }
    return results

# Run the test
stationarity_results = test_stationarity_with_window(data)

# Display results
for node in stationarity_results:
    print(node, stationarity_results[node]["non_stationary_window"])

# Plotting the merged series for a node
node = "m3-123"
all_transmissions = []
for link, transmissions in data[node].items():
    if len(transmissions) >= 90:
        all_transmissions.extend(transmissions)
    
plt.figure(figsize=(12, 6))
plt.plot(all_transmissions)
plt.title(f"Transmissions for Node {node}")
plt.xlabel("Transmission")
plt.ylabel("Transmission Count")

plt.show()