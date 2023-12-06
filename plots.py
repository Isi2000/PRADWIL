import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import powerlaw
from tqdm import tqdm

# Read data for degree distribution
degree_sequence = np.load('./results/degree_sequence.npy')

# Removing the zeros
degree_sequence = degree_sequence[degree_sequence != 0]

# Fit function
fit_function = powerlaw.Fit(degree_sequence, discrete=True)
xmin = fit_function.power_law.xmin  # minimum value to fit
alpha = fit_function.power_law.alpha  # exponent
sigma = fit_function.power_law.sigma  # standard deviation of alpha
D = fit_function.power_law.D  # Kolmogorov-Smirnov statistic (good fit if D is small)

print('\n', f'xmin = {xmin}')
print(f'alpha = {alpha}')
print(f'sigma = {sigma}')
print(f'D = {D}')

# Plotting the degree distribution and fit
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)  # First subplot

plt.yscale('log')
plt.xscale('log')

# Plotting the empirical data with both line and points
data = [x for x in degree_sequence if x > xmin]
fig = powerlaw.plot_pdf(data, color='b', marker='o', linewidth=2, label='Empirical data')

# Plotting the fit of the power law
fit_function.power_law.plot_pdf(ax=fig, color='r', linestyle='--', label='Power law fit')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Degree', fontsize=16)
plt.ylabel('Probability Density', fontsize=16)
plt.title('Degree Distribution with Power Law Fit', fontsize=14)
plt.legend()

# Plotting the cumulative degree distribution and fit
plt.subplot(1, 2, 2)  # Second subplot

plt.yscale('log')
plt.xscale('log')

# Plotting the empirical data with both line and points
data = [x for x in degree_sequence if x > xmin]
fig = powerlaw.plot_cdf(data, color='b', marker='o', linewidth=2, label='Empirical data')

# Plotting the fit of the power law
fit_function.power_law.plot_cdf(ax=fig, color='r', linestyle='--', label='Power law fit')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Degree', fontsize=16)
plt.ylabel('Cumulative Probability', fontsize=16)
plt.title('Cumulative Degree Distribution with Power Law Fit', fontsize=14)
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('tieni.png')
