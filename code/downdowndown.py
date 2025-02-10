import json
import pandas as pd
import os
import requests
from shapely.wkt import loads
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
df = pd.read_csv("/mntssd/mnt3/shanshanbai/nlpinearthobservation/synthetic_data/merged.csv", sep=",")
df['geometry'] = df['geometry'].apply(loads)

# Output JSON Lines file to save results dynamically
output_file = "/mntssd/mnt3/shanshanbai/my_storage_from_qian/results/overpass_results.jsonl"

# Load already processed indices
try:
    with open(output_file, "r") as f:
        processed_indices = {json.loads(line).get('index') for line in f if 'index' in json.loads(line)}
except FileNotFoundError:
    processed_indices = set()

# Define Overpass API endpoint
url = "https://overpass-api.de/api/interpreter"

# Function to decode Unicode escape sequences
def decode_unicode(string):
    try:
        return json.loads(f'"{string}"')
    except json.JSONDecodeError:
        return string

# Function to extract building info
def extract_building_info(relation):
    tags = relation.get('tags', {})
    building_type = tags.get('building')
    name = decode_unicode(tags.get('name', ''))
    operator = decode_unicode(tags.get('operator', ''))
    return building_type, name, operator

# Function to process a single geometry
def process_geometry(index, geometry, tweet_lang):
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    south, west, north, east = bounds[1], bounds[0], bounds[3], bounds[2]

    query = f"""[out:json];
      relation({south},{west},{north},{east})[building];
    out tags;
    (._;>;);
    out qt;
    """

    try:
        response = requests.get(url, params={"data": query}, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Overpass API for geometry {index}: {e}")
        return None

    relations = [element for element in data.get('elements', []) if element['type'] == 'relation']

    combined_data = {
        'index': index,
        'building tags': [],
        'building names': [],
        'building operators': [],
        'tweet language distribution': tweet_lang
    }

    for relation in relations:
        building_type, name, operator = extract_building_info(relation)
        if building_type and building_type != 'yes':
            combined_data['building tags'].append(building_type)
        if name:
            combined_data['building names'].append(name)
        if operator:
            combined_data['building operators'].append(operator)

    # Deduplicate lists
    combined_data['building tags'] = list(set(combined_data['building tags']))
    combined_data['building names'] = list(set(combined_data['building names']))
    combined_data['building operators'] = list(set(combined_data['building operators']))

    return combined_data

# Process geometries
start_time = time.time()
total_geometries = len(df)
results = []
for index, row in df.iterrows():
    if index in processed_indices:
        logging.info(f"Skipping already processed geometry {index}")
        continue

    result = process_geometry(index, row['geometry'], row['tweet_lang'])
    if result:
        results.append(result)

        # Save to file dynamically
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

        elapsed_time = time.time() - start_time
        progress = (index + 1) / total_geometries * 100
        logging.info(f"Saved results for geometry {index} | Progress: {progress:.2f}% | Time Elapsed: {elapsed_time:.2f}s")

logging.info("Processing complete.")
