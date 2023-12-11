import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from tqdm import tqdm

# Read the data

degree_sequence = np.load('./results/degree_sequence_cleaned.npy')

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
plt.figure(figsize=(13, 7))
plt.subplot(1, 2, 1)  # First subplot

plt.yscale('log')
plt.xscale('log')

# Plotting the empirical data 
data = [x for x in degree_sequence if x > xmin]
fig = powerlaw.plot_pdf(data, color='b', linewidth=2, label='Empirical data')

# Plotting the fit of the power law
fit_function.power_law.plot_pdf(ax=fig, color='r', linestyle='--', label='Power law fit')

#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)
plt.xlabel('Degree', fontsize=12, fontweight='bold')
plt.ylabel('Probability Density', fontsize=12, fontweight='bold')
plt.title('Degree Distribution with Power Law Fit', fontsize=14, fontweight='bold')
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

#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)
plt.xlabel('Degree', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
plt.title('Cumulative Degree Distribution with Power Law Fit', fontsize=14, fontweight='bold')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap

#Saving the plot
plt.savefig('./images/degree_distribution_cleaned.png', dpi=300, bbox_inches='tight')

# Calculate n_tail and n
n_tail = np.sum(degree_sequence >= xmin)
n = len(degree_sequence)
head_data = degree_sequence[degree_sequence < xmin]
p_tail = n_tail/n


def generate_dataset(n, p_tail, head_data):
    """
    This function generates a dataset of length n that follows the same distribution of the original dataset.

    Parameters
    ----------
    n : int
        Length of the dataset to generate.
    p_tail : float
        Probability of generating an element from the tail of the distribution.
    head_data : numpy.ndarray
        Head of the original dataset: elements with x < xmin.

    Returns
    -------
    list
        A list of length n that follows the same distribution of the original dataset.
    """
    generated_dataset = []
    for _ in range(n):
        # Genera un numero casuale tra 0 e 1
        p = np.random.rand()

        if p < p_tail:
            # Genera un elemento dalla power law con x > xmin
            generated_value = fit_function.power_law.generate_random(1, estimate_discrete=True)
        else:
            # Pesca un elemento dalla testa del dataset originale con x < xmin
            generated_value = np.random.choice(head_data, 1)
            
        generated_dataset.append(generated_value)
    generated_dataset = np.array(generated_dataset).flatten()
    return generated_dataset

# Number of datasets to generate
num_datasets = 10000

# List to store the D values
D_values = []

# Generate the datasets and calculate the D values
for _ in tqdm(range(num_datasets)):
    generated_dataset = generate_dataset(n, p_tail, head_data)
    fit_function = powerlaw.Fit(generated_dataset, discrete=True)
    D_values.append(fit_function.power_law.D)

# Calculate the p-value

p_value = np.sum(D_values >= D) / num_datasets
print(p_value)

# Plot the D values distribution

plt.figure(figsize=(10, 7))
plt.hist(D_values, bins=50)
plt.xlabel('D value', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('D values distribution', fontsize=18)
plt.savefig('./images/D_values_distribution_cleaned.png')

