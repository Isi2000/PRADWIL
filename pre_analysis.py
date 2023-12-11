import pandas as pd
import os
import analysis_functions as af
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Leggi i dati
print('Reading the data...')
df = pd.read_json("./data/data_cleaned.json")

# Pre-analisi dei dati
df['Year'] = df['Date'].apply(af.convert_year)
print('Number of articles: ', len(df))
print('Min year: ', df['Year'].min())
print('Max year: ', df['Year'].max())

# Aggiungi intervalli di anni
interval_length = 4
df = af.add_year_interval(df, interval_length=interval_length)

# Calcola il numero di articoli e autori per intervallo di anni
print('Computing the number of articles and authors per year interval...')
num_articles_per_interval = df['YearInterval'].value_counts().sort_index()
df['NumAuthors'] = df['Authors'].apply(af.count_authors)
num_authors_per_interval = df.groupby('YearInterval')['NumAuthors'].sum().sort_index()

result_df = pd.DataFrame({
    'YearInterval': num_articles_per_interval.index,
    'NumArticles': num_articles_per_interval.values,
    'NumAuthors': num_authors_per_interval.values
})

# Calcola il numero medio di autori per articolo
result_df['AvgAuthorsPerArticle'] = result_df['NumAuthors'] / result_df['NumArticles']

# Crea la cartella delle immagini se non esiste
os.makedirs('images', exist_ok=True)

# Filtra i dati per il periodo di interesse
filtered_df = result_df.loc[(result_df['YearInterval'] >= 1960) & (result_df['YearInterval'] <= 2020)]

# Creazione di una figura con una griglia di 1 riga e 2 colonne
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Grafico 1: Numero medio di autori per articolo
axs[0].scatter(filtered_df['YearInterval'], filtered_df['AvgAuthorsPerArticle'], s=30, marker='^')
axs[0].set_xlabel('Year interval', fontweight='bold')
axs[0].set_ylabel('Average number of authors per article', fontweight='bold')
axs[0].set_title('Evolution of the average number of authors per paper', fontsize=13, fontweight='bold')
axs[0].legend(['Avg Authors per Article'])  # Aggiungi la legenda al primo grafico
axs[0].grid(True, linestyle='--')  # Aggiungi la griglia al primo grafico

# Grafico 2: Numero di articoli e autori
axs[1].scatter(filtered_df['YearInterval'], filtered_df['NumArticles'], s=30, c='b', marker='v', label='Number of Articles')
axs[1].scatter(filtered_df['YearInterval'], filtered_df['NumAuthors'], s=30, c='r', marker='o', label='Number of Authors')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('Year interval', fontweight='bold')
axs[1].set_ylabel('Number of Articles and Authors', fontweight='bold')
axs[1].set_title('Evolutions of the number of articles and authors', fontsize=13, fontweight='bold')
axs[1].grid(True, linestyle='--')  # Aggiungi la griglia al secondo grafico

# Fit lineare per il numero di articoli
slope_articles, intercept_articles, r_value_articles, p_value_articles, std_err_articles = linregress(filtered_df['YearInterval'], filtered_df['NumArticles'])
fit_line_articles = slope_articles * filtered_df['YearInterval'] + intercept_articles
axs[1].plot(filtered_df['YearInterval'], fit_line_articles, color='b', linestyle='--', linewidth=0.4,
            label=f'Linear Fit (Articles), R²={r_value_articles**2:.2f}')

axs[1].legend(loc='upper left')

# Fit polinomiale per il numero di articoli
degree = 2  # Grado del polinomio
coefficients_articles_poly = np.polyfit(filtered_df['YearInterval'], filtered_df['NumAuthors'], degree)
fit_poly_articles = np.polyval(coefficients_articles_poly, filtered_df['YearInterval'])
# Calcola la media dei dati osservati
mean_observed = np.mean(filtered_df['NumAuthors'])

# Calcola l'R-squared
r_squared_poly_articles = 1 - np.sum((filtered_df['NumAuthors'] - fit_poly_articles)**2) / np.sum((filtered_df['NumAuthors'] - mean_observed)**2)


axs[1].plot(filtered_df['YearInterval'], fit_poly_articles, color='r', linestyle='--', linewidth=0.4,
            label=f'Polynomial Fit (Authors), R²={np.corrcoef(filtered_df["NumAuthors"], fit_poly_articles)[0,1]**2:.2f}')

axs[1].legend(loc='upper left')

plt.tight_layout()

# Salvataggio dell'immagine
plt.savefig('./images/pre_analysis_plots_cleaned.png')

# Stampa i risultati del fit lineare
print(f"Risultati del fit lineare (Articles):")
print(f"Slope: {slope_articles}")
print(f"Intercept: {intercept_articles}")
print(f"R-squared: {r_value_articles**2}")

# Stampa i risultati del fit polinomiale
print(f"Risultati del fit polinomiale (Authors):")
print(f"Coefficients: {coefficients_articles_poly}")
print(f'R-squared (Polynomial Fit - Authors): {r_squared_poly_articles:.2f}')