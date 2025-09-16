import json
import matplotlib.pyplot as plt
import numpy as np

# Input JSON file
input_json_file = "../data/diff-config-long-1-transmissions-per-config-series.json"

# Load data from JSON file
with open(input_json_file, 'r') as infile:
    data = json.load(infile)

# Prepare data for plotting
node_ids = sorted(data.keys(), key=lambda x: int(x.split('-')[1]))

# Collect unique transmission counts and their frequencies for each node
node_frequencies = {}
for node in node_ids:
    node_data = data[node]
    transmissions = []
    for config in node_data.values():
        transmissions.extend(config)
    unique, counts = np.unique(transmissions, return_counts=True)
    
    # Combine transmissions greater than 0.25 into one category
    combined_freqs = {}
    for u, c in zip(unique, counts):
        if u < 0.5 and u>0:
            if 0.24 not in combined_freqs:
                combined_freqs[0.24] = 0
            combined_freqs[0.24] += c
        else:
            combined_freqs[u] = c

    node_frequencies[node] = combined_freqs

# Prepare the data for plotting
unique_transmissions = sorted(set(t for freqs in node_frequencies.values() for t in freqs))
x_positions = np.arange(len(node_ids))  # Base positions for nodes

# Adjust bar width for thicker bars
bar_width = 0.7 / len(unique_transmissions)  # Increased bar width for thicker bars
offsets = np.arange(len(unique_transmissions)) * bar_width - (len(unique_transmissions) * bar_width / 2)

# Plot grouped bar charts
plt.figure(figsize=(16, 8))
for i, t in enumerate(unique_transmissions):
    frequencies = [node_frequencies[node].get(t, 0) for node in node_ids]
    bars = plt.bar(
        x_positions + offsets[i],  # Keep bars centered
        frequencies,
        width=bar_width,
        #label=f"p = {int(t) if isinstance(t, (int, np.integer)) else t:.2g}",
        label = f"2 < # Trans. < 8" if t == 0.24 else (f"# Trans. = {1/t:.2g}" if t != 0 else "Transmission Failure"),
        alpha=0.7,
        edgecolor="black"
    )

    # Annotate each bar with its corresponding value
    for bar in bars:
        height = bar.get_height()
        if height > 10000:  # Only annotate if the bar height is non-zero
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position of the text (centered on the bar)
                height + 2000,  # Y position of the text (slightly above the bar)
                f'{int(height)}',  # Display the value as integer
                ha='center',  # Horizontal alignment: center
                va='bottom',  # Vertical alignment: bottom
                fontsize=13,
                rotation=90
            )
        elif height > 5000:  # Only annotate if the bar height is non-zero
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position of the text (centered on the bar)
                height + 800,  # Y position of the text (slightly above the bar)
                f'{int(height)}',  # Display the value as integer
                ha='center',  # Horizontal alignment: center
                va='bottom',  # Vertical alignment: bottom
                fontsize=13,
                rotation=90
            )
        elif height > 500:  # Only annotate if the bar height is non-zero
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position of the text (centered on the bar)
                height + 50,  # Y position of the text (slightly above the bar)
                f'{int(height)}',  # Display the value as integer
                ha='center',  # Horizontal alignment: center
                va='bottom',  # Vertical alignment: bottom
                fontsize=13,
                rotation=90
            )

font_size = 10
plt.xticks(ticks=x_positions, labels=node_ids, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(loc="upper left", fontsize=font_size)

# Customize the plot
plt.title("Frequency of Transmissions Per Node")
plt.xlabel("Node ID")
plt.ylabel("Frequency")
plt.yscale("log")
plt.ylim(300, 60000)  # Limit the y-axis to better visualize the data
plt.xticks(ticks=x_positions, labels=node_ids)
plt.legend(loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save or show the plot
plt.tight_layout()
plt.savefig("../figures/transmissions_grouped_by_node.pdf", format="pdf", bbox_inches="tight")
plt.show()
