import numpy as np
import matplotlib.pyplot as plt

# Assuming data is loaded into NumPy arrays
eig = np.load("./results/eigenvector_centrality_cleaned.npy")
deg = np.load("./results/degree_centrality_cleaned.npy")
bet = np.load("./results/betweenness_centrality_cleaned.npy")
clo = np.load("./results/closeness_centrality_cleaned.npy")

# Mapping colors to centrality measures
centrality_colors = {
    'eig': 'red',
    'deg': 'green',
    'bet': 'blue',
    'clo': 'orange',
}

# Function to plot scatter plot for a centrality measure
def plot_scatter(ax, data, title, color):
    names = data[:, 0]
    values = data[:, 1].astype(float)

    ax.scatter(names, values, color=color, s=5)  # Adjust the size of points with 's'
    ax.set_title(title)
    ax.set_xticks([])  # Remove x-axis ticks and labels
    ax.set_yticks([])  # Remove y-axis ticks and labels
    if color == 'orange':
        ax.text(names[0], values[0], f"{names[0]}: {values[0]:.4f}", fontsize=8)
    else:
        for i in range(3):
            ax.text(names[i], values[i], f"{names[i]}: {values[i]:.4f}", fontsize=8)
# Create a 2 by 2 grid for subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot scatter plots for each centrality measure
plot_scatter(axs[0, 0], eig, 'Eigenvector Centrality', centrality_colors['eig'])
plot_scatter(axs[0, 1], deg, 'Degree Centrality', centrality_colors['deg'])
plot_scatter(axs[1, 0], bet, 'Betweenness Centrality', centrality_colors['bet'])
plot_scatter(axs[1, 1], clo, 'Closeness Centrality', centrality_colors['clo'])

plt.tight_layout()
plt.savefig('./images/scatter_remondini_cleaned.png')
