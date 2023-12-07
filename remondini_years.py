import pandas as pd

# Assuming df is loaded into a DataFrame
df = pd.read_json("./data/data.json")

print(min(df['Date']))

# Create a dictionary with author names as keys and a list of years as values
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
    auth_max_min_y[i[0]] = (max(i[1]) - min(i[1]))
auth_max_min_y = dict(sorted(auth_max_min_y.items(), key=lambda item: item[1]))
print(auth_max_min_y)


