import subprocess
import csv
import time
from collections import Counter
import json
authors_res = {}
with open("authors.txt", "r") as input_file:
    for query_term in input_file:
        time.sleep(1)
        query = f'{query_term.strip()} [AUTH]'
        command = f'esearch -db pubmed -query "{query}" | efetch -format docsum | xtract -pattern DocumentSummary -element FullJournalName'

        try:
            result = subprocess.check_output(command, shell=True, text=True)
            result = result.strip().split('\n')
            counter = Counter(result)
            authors_res[query_term.strip()] = counter.most_common(3)
            if not result:
                print(f"No result for: {query_term.strip()}")

        except subprocess.CalledProcessError:
            print("Error executing command")

# Save to JSON file
with open("authors_results.json", "w") as json_file:
    json.dump(authors_res, json_file, indent=2)

print("Results saved to authors_results.json")


