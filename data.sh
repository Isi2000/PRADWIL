#!/bin/bash



# Define the query
QUERY="Prader-Willi"

esearch -db pubmed -query ${QUERY} | efetch -format xml | xtract -pattern PubmedArticle -block MedlineCitation -sfx " | "  -element PMID -block Author -sep " " -element LastName,ForeName -block PubDate -pfx " | " -element Year > ids_authors_year.txt

echo "Ids and Authors have been saved to ids_authors_year.txt"
