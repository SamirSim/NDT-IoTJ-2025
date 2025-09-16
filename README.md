# FIT IoT-Lab CSMA Experimentation

This code corresponds to the two contributions described in the following paper (please cite this in case you use the code):

```
@article{si2025config-pred,
  title={Data-Driven Prediction Models for Wireless Network Configuration},
  author={Si-Mohammed, Samir and Théoleyre, Fabrice},
  journal={International Conference on Advanced Information Networking and Applications (AINA)},
  year={2025}
}
```

This project contains firmware, scripts, and analysis tools for running experiments on the FIT IoT-Lab platform, focusing on 802.15.4 CSMA configurations behavior and their impact on the number of packet retransmissions, and how we can provide accurate predictions for unexplored configurations.

## Code Structure

### `code/` - Log Processing & Analysis

- **`preprocess_log.py`**: Converts experiment logs into structured JSON series.
- **`learn-combined-multiple-runs.py`**: Uses the generated series to evaluate prediction accuracy (MAE, MSE) for single-link, global, and k-NN models across various data splits.
- **`plot-results-split-boxplot.py`**: Generates comparative boxplots for different prediction approaches.
- **`plot_results.py`**: Plots model results focusing on the 70% training data split.
- **`print_common_configurations_table.py`**: Analyzes and prints statistics on retransmissions across common configurations.
- **`clusterize_links_ranks.py`**: Clusterizes links according to their performance on similar configurations (the procedure is described in the paper). Return a dict containing the node ids in each cluster.
- **`clusterize_links_pdr.py`**: Clusterizes links according to their mean PRR over all configurations. Return a dict containing the node ids in each cluster.
- **`learn_combined_multiple_runs_clustering.py`**: Performs a training/testing process on the single, merged, k-NN and cluster-based techniques on the provided data. The clusters are specified in the beginning of the script.
- **`test_generalization_clustering(_online).py`**: Tests the trained models with the data from each cluster on additional nodes (not existent in the original experiment) to evaluate the generalization precision. The online version consists in constantly refitting the model with data from the new link.
- **`plot_results_clustering.py`**: Plot the obtained results for the cluster-based techniques.


### `code/expe/` - Experiment Firmware & Execution Scripts

- **`unicast-sender.c`**: Firmware for the sender node (M3) in FIT IoT-Lab.
- **`unicast-receiver.c`**: Firmware for the receiver node (M3) in FIT IoT-Lab.
- **`csma.c`**: CSMA implementation for Contiki-NG, modified to log retransmissions when `DEBUG=1`.
- **`exp-test-config.sh`**: Creates and configures an experiment, assigning firmware and parameters.
- **`run-exp.sh`**: Calls `exp-test-config.sh` with experiment parameters such as packet size.

### `data/` - Experiment Logs

- **`AINA/diff-config-long-*.txt`**: Raw logs from FIT IoT-Lab experiments, later processed for analysis.
- **`IoTJ/expe-iotj-1ps-*.txt`**: Raw logs from FIT IoT-Lab experiments for the IoTJ paper, with 1 packet per second, later processed for analysis.
- **`IoTJ/5ps/expe-iotj-*.txt`**: Raw logs from FIT IoT-Lab experiments for the IoTJ paper, with 5 packets per second, later processed for analysis (not shown in the paper).


## Usage

1. **Run the Experiment**  
   - Configure and launch an experiment with `run-exp.sh`, specifying parameters.
   - Firmware will execute on M3 nodes, logging transmissions and retransmissions.

2. **Process Logs**  
   - Convert logs into structured JSON using `preprocess_log.py`.

3. **Analyze Results**  
   - Train predictive models using `learn-combined-multiple-runs.py`.
   - Generate visualizations with `plot-results-split-boxplot.py` or `plot_results.py`.
   - Extract retransmission statistics with `print_common_configurations_table.py`.

## Dependencies

- **Contiki-NG** for CSMA experiments
- **FIT IoT-Lab** for deployment
- **Python (with NumPy, Pandas, Matplotlib)** for analysis