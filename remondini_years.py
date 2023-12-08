import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


df = pd.read_json("./data/data.json")

"""
author_indices = {}
auths = []
# Iterate through the DataFrame
for index, row in df.iterrows():
    authors = row['Authors']
    for author in authors:
        # Remove leading/trailing spaces
        auths.append(author)
auths = set(auths)
auths = list(auths)
res = {}
for auth in auths:
    if len(auth.split()) != 0:
        
        first_name = auth.split()[0]
        if first_name not in res.keys():
            res[first_name] = [auth]
        else:
            res[first_name].append(auth)
f_res = [(key, value) for key, value in res.items() if len(value) != 1]

print(f_res)
print(len(f_res))



"""
author_years_dict = {}

for index, row in df.iterrows():
    authors = row['Authors']
    year = row['Date']

    for author in authors:
        if author in author_years_dict:
            author_years_dict[author].append(year)
            
        else:
            author_years_dict[author] = [year]


auth_max_min_y = {}
for i in author_years_dict.items():
    auth_max_min_y[i[0]] = (max(i[1]) -  min(i[1]))
auth_max_min_y = dict(sorted(auth_max_min_y.items(), key=lambda item: item[1], reverse = True))
print(auth_max_min_y)
print(len(auth_max_min_y.keys()))
print(auth_max_min_y['Grugni G'])
print(auth_max_min_y['Grugni Graziano'])
eig = np.load("./results/eigenvector_centrality.npy")
deg = np.load("./results/degree_centrality.npy")
bet = np.load("./results/betweeness_centrality.npy")
clo = np.load("./results/closeness_centrality.npy")
eig_positions = {author: position for position, (author, _) in enumerate(eig, start=1)}
deg_positions = {author: position for position, (author, _) in enumerate(deg, start=1)}
bet_positions = {author: position for position, (author, _) in enumerate(bet, start=1)}
clo_positions = {author: position for position, (author, _) in enumerate(clo, start=1)}

p_list = [eig_positions, deg_positions, bet_positions, clo_positions]
r_list = []
for i in p_list:
    auth_in_y = []
    for auth in i.keys():
        auth_in_y.append([auth, auth_max_min_y[auth]])
        r_list.append(auth_in_y)


borda = np.load("./results/borda.npy")
borda_positions = {author: position for position, (author, _) in enumerate(borda, start=1)}
print(len(borda_positions))
"""
# Mapping colors to centrality measures
centrality_colors = {
    'eig': 'red',
    'deg': 'green',
    'bet': 'blue',
    'clo': 'orange',
}

# Function to plot scatter plot for a centrality measure
def plot_scatter(ax, data, title, color):
    names = [item[0] for item in data]
    values = [item[1] for item in data]

    ax.scatter(names, values, color=color, s=5)  # Adjust the size of points with 's'
    ax.set_xlabel("Authors ordered by centrality measure")
    ax.set_ylabel("Time span between most recent and oldest publication")
    ax.set_title(title)
    ax.set_xticks([])  # Remove x-axis ticks and labels
    ax.set_yticks([])  # Remove y-axis ticks and labels
    
# Create a 2 by 2 grid for subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot scatter plots for each centrality measure
plot_scatter(axs[0, 0], r_list[0], 'Eigenvector Centrality', centrality_colors['eig'])
plot_scatter(axs[0, 1], r_list[1], 'Degree Centrality', centrality_colors['deg'])
plot_scatter(axs[1, 0], r_list[2], 'Betweenness Centrality', centrality_colors['bet'])
plot_scatter(axs[1, 1], r_list[3], 'Closeness Centrality', centrality_colors['clo'])

plt.tight_layout()
plt.savefig("scatter_timespan.png")
plt.show()
"""
# Mapping colors to centrality measures
centrality_colors = {
    'eig': 'red',
    'deg': 'green',
    'bet': 'blue',
    'clo': 'orange',
}

borda = np.load("./results/borda.npy")
borda_positions = {author: position for position, (author, _) in enumerate(borda, start=1)}
print(borda)
# Add Borda Centrality positions to the positions list
borda_positions_list = []
for auth in borda_positions.keys():
    borda_positions_list.append([auth, auth_max_min_y[auth]])
r_list.append(borda_positions_list)
# Function to plot histogram for a centrality measure
def plot_histogram(ax, data, title, color):
    print(len(data))
    values = [item[1] for item in data]
    names = [item[0] for item in data]
    ax.plot(names, values) #color=color, edgecolor=color)
    ax.set_title(title)
    ax.set_xlabel("Authors")
    ax.set_ylabel("Time Span (Years)")
    ax.set_xticks([])  # Remove x-axis ticks and labels
    #ax.set_yticks([])  # Remove y-axis ticks and labels

# Create a 2 by 2 grid for subplots
fig, ax = plt.subplots(figsize=(10, 8))

# Plot histogram for Borda Centrality
plot_histogram(ax, r_list[4], 'Borda Score', 'purple')

plt.tight_layout()
#plt.savefig("ultima_figura.png")
plt.show()
"""
