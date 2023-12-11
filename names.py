import pandas as pd
from unidecode import unidecode
import Levenshtein
import json


df = pd.read_json("./data/data.json")

# Functions for cleaning data

df["Authors"] = df["Authors"].apply(lambda authors: [unidecode(author).replace( "  ", " ").replace(".","").replace(",", "").replace("-", " ") for author in authors])
import re

def only_initials(name):
    """
    This function returns True if the name is composed only by initials, False otherwise.
    """
    if re.match(r'^[A-Z\. ]+$', name):
        return True
    else:
        return False


authors_list = [] # List of authors
for authors in df['Authors']:
    for author in authors:
        author = unidecode(author) # Remove accents
        author = author.replace("  ", " ") # Remove double spaces
        author = author.replace(".", "") # Remove dots
        author = author.replace(",", "") # Remove commas
        author = author.replace("-", " ") # Remove dashes and replace them with spaces
        authors_list.append(author)

authors_list = set(authors_list) # Remove duplicates
filtered_authors_list = [author for author in authors_list if len(author.split()) > 0]
#print(len(filtered_authors_list))

last_name_dict = {} # Dictionary of authors with last name as key and list of authors as value

for author in filtered_authors_list:
    last_name = author.split()[0]
    first_name_only = ' '.join(author.split()[1:])
    if last_name not in last_name_dict.keys():
        last_name_dict[last_name] = [first_name_only]
    else:
        last_name_dict[last_name].append(first_name_only)


filtered_last_name_dict = {key: value for key, value in last_name_dict.items() if len(value) > 1}


for key in filtered_last_name_dict.keys():
  filtered_last_name_dict[key] = sorted(filtered_last_name_dict[key])


def is_shortened_author(author1, author2):
    # Calculate Levenshtein distance between author names
    distance = Levenshtein.distance(author1, author2)

    # Define a threshold for similarity (you can adjust this as needed)
    similarity_threshold = 1

    # Check if the Levenshtein distance is below the threshold
    if distance <= similarity_threshold:
        return True

    # Check if one author's name is a substring of the other
    if author1 in author2 or author2 in author1:
        return True

    return False

# Your dictionary

potential_matches_dict = {}


for key, authors in filtered_last_name_dict.items():
    potential_matches = []
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            if is_shortened_author(authors[i], authors[j]):
                potential_matches.append((authors[i], authors[j]))
    if potential_matches:
      for i in potential_matches:
        for word in i:
          potential_matches_dict[key + ' ' + word] = key + ' ' + max(i, key = len)


# Print the dictionary with potential matches within each group


aaa = []
for authors_list in df['Authors']:
    alist = []
    for author in authors_list:
      if author in potential_matches_dict.keys():
        author = potential_matches_dict[author]
      alist.append((author))
    aaa.append(alist)

df["Cleaned_authors"] = aaa

for index, row in df.iterrows():
    authors = row["Authors"]
    cleaned_authors = row["Cleaned_authors"]
    
    # Ensure the lengths of both lists match
    if len(authors) != len(cleaned_authors):
        print(f"Difference in row {index}: Length mismatch")

    for author, cleaned_author in zip(authors, cleaned_authors):
        if author != cleaned_author:
            print(f"Difference in row {index}:")
            print(f"Original Author: {author}")
            print(f"Cleaned Author: {cleaned_author}")
            print("\n")

# Save the cleaned dataframe in json format

df.to_json("./data/data_cleaned.json")

"""
with open('madonna1.txt', 'w') as file:
    for index, row in df.iterrows():
        file.write("Authors: " + str(row["Authors"]) + "\n")
        file.write("Cleaned_authors: " + str(row["Cleaned_authors"]) + "\n")
"""
