import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, linregress
from matplotlib.ticker import ScalarFormatter


df = pd.read_json("./data/data_cleaned.json")

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

print('The authors with the longest activity is: ', list(auth_max_min_y.items())[1], 'with ', list(auth_max_min_y.items())[1][1], 'years of activity')

eig = np.load("./results/eigenvector_centrality_cleaned.npy")
deg = np.load("./results/degree_centrality_cleaned.npy")
bet = np.load("./results/betweenness_centrality_cleaned.npy")
clo = np.load("./results/closeness_centrality_cleaned.npy")
borda = np.load("./results/borda_cleaned.npy")
print(borda)

df_auth_years = pd.DataFrame(list(auth_max_min_y.items()), columns=['Author', 'Years'])

df_eig = pd.DataFrame(eig, columns=['Author', 'Eigenvector Centrality'])
df_deg = pd.DataFrame(deg, columns=['Author', 'Degree Centrality'])
df_bet = pd.DataFrame(bet, columns=['Author', 'Betweenness Centrality'])
df_clo = pd.DataFrame(clo, columns=['Author', 'Closeness Centrality'])
#print(df_eig)
df_borda = pd.DataFrame(borda, columns=['Author', 'Borda Score'])
print(df_borda)

# Unione dei DataFrame basata sulle colonne degli autori
df_combined = pd.merge(df_auth_years, df_eig, on='Author', how='left')
df_combined = pd.merge(df_combined, df_deg, on='Author', how='left')
df_combined = pd.merge(df_combined, df_bet, on='Author', how='left')
df_combined = pd.merge(df_combined, df_clo, on='Author', how='left')
df_combined = pd.merge(df_combined, df_borda, on='Author', how='left')

# Lista delle colonne di misure di centralità
centrality_columns = ['Eigenvector Centrality', 'Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Borda Score']

df_combined = df_combined.dropna(subset=centrality_columns, how='any')

df_combined.reset_index(drop=True, inplace=True)

# Converti le colonne 'Years' e centralità in tipo numerico
for column in centrality_columns:
    df_combined[column] = pd.to_numeric(df_combined[column], errors='coerce')

#Computing the mean of each centrality measure for each year
df_combined['Years'] = df_combined['Years'].astype(int)
df_mean = df_combined.groupby('Years', as_index=False).mean()
print(df_mean)

#Histogram of the years of activity
plt.figure(figsize=(8, 5))
plt.hist(df_combined['Years'], bins=45, log=True)
plt.title('Histogram of the years of activity', fontweight='bold')
plt.xlabel('Years of activity', fontsize = 10 , fontweight='bold')
plt.ylabel('Number of authors', fontsize= 10, fontweight='bold')
plt.xticks(np.arange(0, 45, 5))
plt.tight_layout()
plt.savefig('./images/histogram_years_cleaned.png')

# Creazione della figura e degli assi con 2 righe e 2 colonne
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

# Lista delle colonne di misure di centralità
centrality_columns = ['Eigenvector Centrality', 'Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality']

colors = ['red', 'green', 'blue', 'purple']
mean_colors = ['green', 'purple', 'yellow', 'cyan']

# Loop sugli assi e colonne per creare gli scatter plot
for i, ax in enumerate(axes.flatten()):
    column_name = centrality_columns[i]
    color = colors[i]
    mean_color = mean_colors[i]

    # Scatter plot per la centrality corrente
    ax.scatter(df_combined['Years'], df_combined[column_name], s=5, color=color)
    ax.scatter(df_mean['Years'], df_mean[column_name], s=5, color = mean_color, marker = '^')
    ax.set_title(column_name, fontweight='bold')
    ax.set_xlabel('Years of activity')
    ax.set_ylabel(f'{column_name} value')

    # Calcola il coefficiente di correlazione e p-value
    correlation_coefficient, p_value = pearsonr(df_combined['Years'], df_combined[column_name])
 
    # Esegui la regressione lineare
    slope, intercept, r_value, p_value_regression, std_err = linregress(df_combined['Years'], df_combined[column_name])

    # Calcola i valori della retta di regressione lineare
    regression_line = slope * df_mean['Years'] + intercept
   
    # Imposta le coordinate del testo
    text_x, text_y = 0.05, 0.95
    vertical = 'top'

    if 'Closeness' in column_name:
        text_x, text_y = 0.68, 0.05  # Cambia le coordinate per la centrality specifica
        vertical = 'bottom'

    # Aggiunta del testo con il coefficiente di correlazione e p-value
    text_str = f"Pearson r = {correlation_coefficient:.2f}\np-value = {p_value:.2e}"
    ax.text(text_x, text_y, text_str, transform=ax.transAxes, fontsize=8, verticalalignment=vertical)

    # Aggiungi la retta di regressione lineare
    ax.plot(df_mean['Years'], regression_line, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('./images/scatter_pearson_cleaned.png')

# Scatter plot per 'Borda Score'
plt.figure(figsize=(8, 5))
plt.scatter(df_combined['Years'], df_combined['Borda Score'], color='orange', s=5)
plt.scatter(df_mean['Years'], df_mean['Borda Score'], color='blue', s=8, marker = '^')
correlation_coefficient, p_value = pearsonr(df_combined['Years'], df_combined['Borda Score'])
text_str = f"Pearson r = {correlation_coefficient:.2f}\np-value = {p_value:.2e}"
plt.text(0.95, 0.05, text_str, transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
plt.title('Borda Score', fontweight='bold' )
plt.xlabel('Years of activity', fontsize = 10 , fontweight='bold')
plt.ylabel('Borda Score value', fontsize= 10, fontweight='bold')
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
plt.gca().yaxis.set_major_formatter(formatter)
plt.xticks(np.arange(0, 45, 5))
plt.tight_layout()
plt.savefig('./images/borda_score_cleaned.png')

centrality_columns = ['Degree Centrality', 'Eigenvector Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Borda Score']
df_top_50_subsets = {}

for centrality_column in centrality_columns:
    # Ottieni il nome del DataFrame e della colonna
    df_name = f'df_top_50_{centrality_column.lower().replace(" ", "_")}'
    column_name = centrality_column
    
    # Ordina il DataFrame in base alla colonna corrente in ordine decrescente
    df_top_50 = df_combined.sort_values(by=column_name, ascending=False).head(50)
    
    # Seleziona solo le colonne desiderate
    df_top_50_subset = df_top_50[['Years', column_name]]
    
    # Salva il DataFrame ottenuto nel dizionario
    df_top_50_subsets[df_name] = df_top_50_subset

# Lista delle colonne di misure di centralità (tralasciando 'Borda Score')
centrality_columns = ['Eigenvector Centrality', 'Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality']

# Creazione della figura e degli assi con 2 righe e 2 colonne
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

colors = ['red', 'green', 'blue', 'purple']
mean_colors = ['green', 'purple', 'yellow', 'cyan']

# Loop sugli assi e colonne per creare gli scatter plot
for i, ax in enumerate(axes.flatten()):
    column_name = centrality_columns[i]
    color = colors[i]
    mean_color = mean_colors[i]

    year_values = df_top_50_subsets[f'df_top_50_{column_name.lower().replace(" ", "_")}']['Years']
    centrality_values = df_top_50_subsets[f'df_top_50_{column_name.lower().replace(" ", "_")}'][column_name]

    # Scatter plot per la centrality corrente (solo i 50 autori più influenti)
    ax.scatter(year_values,
               centrality_values,
               s=5, color=color)

    ax.set_title(column_name, fontweight='bold')
    ax.set_xlabel('Years of activity')
    ax.set_ylabel(f'{column_name} value')

    # Calcola il coefficiente di correlazione e p-value
    correlation_coefficient, p_value = pearsonr(year_values, centrality_values)

    # Esegui la regressione lineare per tutti gli autori (non solo i primi 50)
    slope, intercept, r_value, p_value_regression, std_err = linregress(year_values, centrality_values)

    # Calcola i valori della retta di regressione lineare
    regression_line = slope * df_mean['Years'] + intercept

    # Imposta le coordinate del testo
    text_x, text_y = 0.05, 0.95
    vertical = 'top'

    if 'Closeness' in column_name:
        text_x, text_y = 0.68, 0.05  # Cambia le coordinate per la centrality specifica
        vertical = 'bottom'

    # Aggiunta del testo con il coefficiente di correlazione e p-value
    text_str = f"Pearson r = {correlation_coefficient:.2f}\np-value = {p_value:.2e}"
    ax.text(text_x, text_y, text_str, transform=ax.transAxes, fontsize=8, verticalalignment=vertical)

    # Aggiungi la retta di regressione lineare
    ax.plot(df_mean['Years'], regression_line, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('./images/centrality_vs_longevity.png')

# Scatter plot per 'Borda Score'

year_values = df_top_50_subsets['df_top_50_borda_score']['Years']
borda_values = df_top_50_subsets['df_top_50_borda_score']['Borda Score']

plt.figure(figsize=(8, 5))
plt.scatter(year_values, borda_values, color='orange', s=5)
correlation_coefficient, p_value = pearsonr(year_values, borda_values)
text_str = f"Pearson r = {correlation_coefficient:.2f}\np-value = {p_value:.2e}"
plt.text(0.95, 0.05, text_str, transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
plt.title('Borda Score', fontweight='bold' )
plt.xlabel('Years of activity', fontsize = 10 , fontweight='bold')
plt.ylabel('Borda Score value', fontsize= 10, fontweight='bold')
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.savefig('./images/borda_score_top50_cleaned.png')





