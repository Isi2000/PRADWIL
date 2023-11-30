import json

# Assuming the dataset is stored in a file named 'dataset.txt'
dataset_file = 'ids_authors_year.txt'

# Create an empty list to store the data
data_list = []

# Read the dataset file and process each line
with open(dataset_file, 'r') as file:
    for line in file:
        # Split the line into columns using any whitespace characters
        columns = line.strip().split("|")
        
        if (len(columns)) == 3:
            id = columns[0].strip()
            authors = columns[1].strip().split("\t")
            date = int(columns[2].strip())
        else:
            print("bad_data")
        # Create a dictionary for each entry
        entry = {'Id': id, 'Authors': authors, 'Date': date}
        print(entry)
        # Append the entry to the data list
        data_list.append(entry)

# Create a JSON file with the processed data
json_file = 'data.json'
with open(json_file, 'w') as outfile:
    json.dump(data_list, outfile, indent=4)

print(f"JSON file '{json_file}' has been generated successfully.")
