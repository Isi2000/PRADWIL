import pandas as pd
import os
import analysis_functions as af
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Read the data--------------------------------------------------------------
print('Reading the data...')

df = pd.read_json("./data/data.json")

# Pre analysis of the data-------------------------------------------------

df['Year'] = df['Date'].apply(af.convert_year)
print('Number of articles: ', len(df))
print('Min year: ', df['Year'].min())
print('Max year: ', df['Year'].max())

# Print the number of articles and authors

interval_length = 4
df = af.add_year_interval(df, interval_length = interval_length)

# Computing the number of articles and authors per year interval
print('Computing the number of articles and authors per year interval...')

num_articles_per_interval = df['YearInterval'].value_counts().sort_index()

df['NumAuthors'] = df['Authors'].apply(af.count_authors)
num_authors_per_interval = df.groupby('YearInterval')['NumAuthors'].sum().sort_index()

result_df = pd.DataFrame({
    'YearInterval': num_articles_per_interval.index,
    'NumArticles': num_articles_per_interval.values,
    'NumAuthors': num_authors_per_interval.values
})

# Computing the average number of authors per article
result_df['AvgAuthorsPerArticle'] = result_df['NumAuthors'] / result_df['NumArticles']

# Saving the results in a JSON file

#os.makedirs('./results', exist_ok=True)

#print('Saving the results...')
#result_df.to_json('./results/num_paper_authors.json', orient='records', lines=True, index = True)
#print('Results saved in the results folder!')

# Plotting the results

os.makedirs('images', exist_ok=True)

filtered_df = result_df.loc[(result_df['YearInterval'] >= 1960) 
                            & (result_df['YearInterval'] <= 2020)]

# Plotting the results

os.makedirs('images', exist_ok=True)

# Plotting the results side by side

import matplotlib.pyplot as plt


# Creazione di una figura con una griglia di 1 riga e 2 colonne
fig, axs = plt.subplots(1, 2, figsize=(13, 6))

# Primo grafico
axs[0].scatter(filtered_df['YearInterval'], filtered_df['AvgAuthorsPerArticle'], s = 30, marker = '^')
axs[0].set_xlabel('Year interval', fontweight='bold')
axs[0].set_ylabel('Average number of authors per article', fontweight='bold')
axs[0].set_title('Evolution of the average number of authors per paper', fontsize=13, fontweight='bold')
axs[0].legend(['Avg Authors per Article'])  # Aggiungi la legenda al primo grafico
axs[0].grid(True, linestyle = '--')  # Aggiungi la griglia al primo grafico

# Secondo grafico
axs[1].scatter(filtered_df['YearInterval'], filtered_df['NumArticles'], s=30, c='b', marker='v', label='Number of Articles')
axs[1].scatter(filtered_df['YearInterval'], filtered_df['NumAuthors'], s=30, c='r', marker='o', label='Number of Authors')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('Year interval', fontweight='bold')
axs[1].set_ylabel('Number of Articles and Authors', fontweight='bold')
axs[1].set_title('Evolutions of the number of articles and authors', fontsize=13, fontweight='bold')
axs[1].grid(True, linestyle = '--')  # Aggiungi la griglia al secondo grafico

# Fit lineare per il numero di articoli
slope, intercept, r_value, p_value, std_err = linregress(filtered_df['YearInterval'], filtered_df['NumArticles'])
fit_line_articles = slope * filtered_df['YearInterval'] + intercept
axs[1].plot(filtered_df['YearInterval'], fit_line_articles, color='b', linestyle='--', linewidth = 0.3, label=f'Linear Fit (Articles), R²={r_value**2:.2f}')

axs[1].legend(loc='upper left')

# Imposta i segnaposto degli intervalli e le etichette sull'asse x
interval_ticks = np.arange(filtered_df['YearInterval'].min(), filtered_df['YearInterval'].max() + 1, interval_length)

axs[1].set_xticks(interval_ticks)
axs[1].set_xticklabels([f'{year}-{year+interval_length-1}' for year in interval_ticks], rotation=45, ha='right', fontsize = 6)
axs[0].set_xticks(interval_ticks)
axs[0].set_xticklabels([f'{year}-{year+interval_length-1}' for year in interval_ticks], rotation=45, ha='right', fontsize = 6)

# Fit esponenziale per il numero di autori
#def exponential_fit(x, a, b, c):
#    return a * np.exp(b * x) + c

#params_authors, covariance_authors = curve_fit(exponential_fit, filtered_df['YearInterval'], filtered_df['NumAuthors'])
#fit_exponential_authors = exponential_fit(filtered_df['YearInterval'], *params_authors)
#axs[1].plot(filtered_df['YearInterval'], fit_exponential_authors, color='r', linestyle='--', label=f'Exponential Fit (Authors), R²={np.corrcoef(filtered_df["NumAuthors"], fit_exponential_authors)[0,1]**2:.2f}')

#axs[1].legend(loc='upper left')

plt.tight_layout()

# Salvataggio dell'immagine
plt.savefig('./images/pre_analysis_plots.png')

