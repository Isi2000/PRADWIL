#!/bin/bash

# Define the query
QUERY="Prader-Willi"

# Use esearch to retrieve the ids
IDS=$(esearch -db pubmed -query "$QUERY" | efetch -format docsum | xtract -pattern Id -element Id)

# Use efetch to retrieve the authors for each id
echo "$IDS" | while read -r ID; do
  AUTHORS=$(efetch -db pubmed -id "$ID" -format docsum | xtract -pattern AuthList -element Author | tr '\n' ',' | sed 's/,$//')
  echo "$ID,$AUTHORS"
done > ids_and_authors.txt

echo "Ids and Authors have been saved to ids_and_authors.txt"
