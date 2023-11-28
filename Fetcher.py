import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time


def get_pubmed_ids(term, retstart, retmax):
    """
    Fetch PubMed IDs for a specific term.

    Args:
        term (str): The search term.
        retstart (int): The index of the first record to retrieve.
        retmax (int, optional): The maximum number of records to retrieve. Default is 50.

    Returns:
        list: List of PubMed IDs.
    """
    esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retstart={retstart}&retmax={retmax}&format=json&sort=relevance"
    response = requests.get(esearch_url)
    ids = response.json()["esearchresult"]["idlist"]
    return ids

def parse_article(article):
    """
    Parse individual PubMed article data from XML.

    Args:
        article (xml.etree.ElementTree.Element): XML element for a single article.

    Returns:
        dict: Parsed article data.
    """
    # Extract PubMed ID
    pubmed_id = article.find(".//MedlineCitation/PMID").text

    # Extract publication date
    pub_date_elem = article.find(".//PubDate")
    year = pub_date_elem.findtext(".//Year") or "Unknown"
    month = pub_date_elem.findtext(".//Month") or "Unknown"
    day = pub_date_elem.findtext(".//Day") or "Unknown"
    pub_date = f"{year}-{month}-{day}"

    # Extract authors
    authors_elem = article.find(".//AuthorList")
    authors = [
        f"{author.findtext('.//LastName') or author.findtext('.//CollectiveName') or 'pd'} {author.findtext('.//ForeName') or author.findtext('.//CollectiveName') or 'pd'}"
        for author in authors_elem.findall(".//Author")
    ] if authors_elem is not None else ["Unknown"]

    # Extract MeSH terms with MajorTopicYN distinction
    mesh_elems = article.findall(".//MeshHeadingList/MeshHeading")
    mesh_terms = []
    for mesh in mesh_elems:
        descriptor = mesh.find('.//DescriptorName')
        descriptor_name = descriptor.text
        descriptor_major_topic = descriptor.get('MajorTopicYN') == 'Y'

        # Process QualifierNames if any
        qualifiers = mesh.findall('.//QualifierName')
        for qual in qualifiers:
            qual_name = qual.text
            qual_major_topic = qual.get('MajorTopicYN') == 'Y'
            mesh_terms.append({
                'DescriptorName': descriptor_name,
                'DescriptorMajorTopic': descriptor_major_topic,
                'QualifierName': qual_name,
                'QualifierMajorTopic': qual_major_topic
            })

        # Include DescriptorName if there are no QualifierNames
        if not qualifiers:
            mesh_terms.append({
                'DescriptorName': descriptor_name,
                'DescriptorMajorTopic': descriptor_major_topic,
                'QualifierName': None,
                'QualifierMajorTopic': None
            })

    return {
        "Id": pubmed_id,
        "Dates": pub_date,
        "Authors": authors,
        "MeSH_Terms": mesh_terms
    }

def fetch_data_for_ids(ids, retries=5, delay=0.5):
    for attempt in range(retries):
        id_string = ",".join(str(id) for id in ids)
        efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_string}&rettype=abstract"
        response = requests.get(efetch_url)
        if response.status_code == 200:
            try:
                root = ET.fromstring(response.text)
                return [parse_article(article) for article in root.findall(".//PubmedArticle")]
            except ET.ParseError as e:
                print(f"Error parsing XML for IDs: {ids}")
                print(f"Error details: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred for IDs: {ids}")
                print(f"Error details: {e}")    
                break
            time.sleep(delay)
        else:
            print(f"Failed to fetch data for IDs: {ids}. Status Code: {response.status_code}")
            print(f"Failed XML:\n{response.text}")
            time.sleep(delay)
            porcodio.append(ids)
            #np.savetxt(f"./DATA/{ids[0]}.csv", ids, delimiter=",", fmt='%s')
            break

    return []

def fetch_pubmed(ids_list):
    """
    Fetch PubMed data sequentially for multiple lists of IDs.

    Args:
        ids_list (list of list): List containing sublists of PubMed IDs.

    Returns:
        pd.DataFrame: DataFrame containing combined PubMed data.
    """
    all_results = []

    for ids in ids_list:
        fetched_data = fetch_data_for_ids(ids)
        all_results.extend(fetched_data)

    return pd.DataFrame(all_results)


# Usage
retmax = 50
data = []
for i in tqdm(range(100)):
    ids = get_pubmed_ids("Prader Willi", retstart=i * retmax, retmax=retmax)
    print(ids)
    df = fetch_pubmed(ids)
    df.to_json(f"./DATA/pubmed_data_{i}_nuovi.json", orient="records", lines=True)
    data.append(df)
#print(data)
unique_df = pd.concat(data)

unique_df.to_json("pubmed_data.json", orient="records", lines=True)

    
