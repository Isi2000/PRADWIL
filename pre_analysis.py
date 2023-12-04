import pandas as pd
import os
import analysis_functions as af

# Read the data--------------------------------------------------------------
print('Reading the data...')

df = pd.read_json("data.json")

# Pre analysis of the data-------------------------------------------------

df['Year'] = df['Date'].apply(af.convert_year)

df = af.add_year_interval(df, interval_length = 5)

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

os.makedirs('./results', exist_ok=True)

print('Saving the results...')
result_df.to_json('./results/num_paper_authors.json', orient='records', lines=True, index = True)
print('Results saved in the results folder!')