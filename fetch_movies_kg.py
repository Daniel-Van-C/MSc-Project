# fetch_movies_kg.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Retrieves a dataset of movies knowledge graph properties from Wikidata.
# 

# Imports.
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

def get_movie_data(movie_title):
    """Get the movie details from wikidata.

    This function retrieves details of a movie using the SPARQL endpoint
    of wikidata. The retrieved details include cast, director, producer,
    screenwriter, composer, editor, distributor, and country. If the request
    fails, the function will keep retrying until it succeeds.

    Args:
        movie_title: A string that represents the title of the movie.

    Returns:
        A dictionary that includes all retrieved details of the movie.

    """
    time.sleep(1)
    global movie_index, error_list
    movie_index += 1
    
    execute = True
    while execute:
        print("Movie number:", movie_index)
        try:
            # Set up SPARQL query
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setQuery(f"""
                SELECT 
                    ?castLabel ?directorLabel ?producerLabel ?screenwriterLabel ?composerLabel ?editorLabel ?year ?distributorLabel ?countryLabel
                WHERE {{
                    ?film wdt:P31 wd:Q11424 .
                    ?film rdfs:label ?label .
                    FILTER(LANG(?label) = "en") .
                    FILTER(CONTAINS(?label, "{movie_title}")) .
                    OPTIONAL {{ ?film wdt:P161 ?cast . }}
                    OPTIONAL {{ ?film wdt:P57 ?director . }}
                    OPTIONAL {{ ?film wdt:P162 ?producer . }}
                    OPTIONAL {{ ?film wdt:P58 ?screenwriter . }}
                    OPTIONAL {{ ?film wdt:P86 ?composer . }}
                    OPTIONAL {{ ?film wdt:P1040 ?editor . }}
                    OPTIONAL {{ ?film wdt:P577 ?year . }}
                    OPTIONAL {{ ?film wdt:P750 ?distributor . }}
                    OPTIONAL {{ ?film wdt:P495 ?country . }}
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
            """)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return results['results']['bindings']
        
        except Exception as e:
            # Log the error
            error_list.append(str(movie_index) + e)
            print("An error occurred:", e)
            time.sleep(5)
            if e != "HTTP Error 429: Too Many Requests":
                return None

def update_df(idx):
    """Updates the DataFrame with the retrieved movie knowledge graph
       properties.

    Args:
        idx: An integer that represents the index of the row in the DataFrame
             to be updated.
    """
    global df
    data = get_movie_data(df.loc[idx, 'title'])
    if data:  # if data exists
        for item in data:
            # Update the respective columns in the dataframe
            if 'castLabel' in item:
                df.loc[idx, 'cast'] = item['castLabel']['value']
            if 'directorLabel' in item:
                df.loc[idx, 'director'] = item['directorLabel']['value']
            if 'producerLabel' in item:
                df.loc[idx, 'producer'] = item['producerLabel']['value']
            if 'screenwriterLabel' in item:
                df.loc[idx, 'screenwriter'] = item['screenwriterLabel']['value']
            if 'composerLabel' in item:
                df.loc[idx, 'composer'] = item['composerLabel']['value']
            if 'editorLabel' in item:
                df.loc[idx, 'editor'] = item['editorLabel']['value']
            if 'distributorLabel' in item:
                df.loc[idx, 'distributor'] = item['distributorLabel']['value']
            if 'countryLabel' in item:
                df.loc[idx, 'country'] = item['countryLabel']['value']

def main():
    # Creates 'Data' directory if it doesn't exist.
    if not os.path.exists("Data"):
        os.makedirs("Data")
    # Global variables that multiple concurrent threads can access.
    global df, movie_index, error_list
    movie_index = 0
    error_list = []
    # Loads movies dataset.
    df = pd.read_csv(os.path.join("Data", 'movies.csv')).head(4)
    # Removes movies with no genres assigned.
    df['genres'] = df['genres'].replace({'(no genres listed)': np.nan})
    df = df.dropna(subset=['genres'])
    # Extracts and removes release year from title.
    df['year'] = df['title'].apply(lambda x: x[-5:-1])
    df['title'] = df['title'].apply(lambda x: x[:-7])
    # Uses multithreading to improve runtime.
    completed_tasks = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(update_df, idx): idx for idx in df.index}
        for future in as_completed(futures):
            completed_tasks += 1
            
    # Saves movies knowledge graph to CSV file.
    df.to_csv(os.path.join("Data", 'movies_kg_full.csv'), index=False)
    # Saves any errors into a text file.
    with open(os.path.join("Data", 'wikidata_error_file.txt'), 'w') as file:
        for item in error_list:
            file.write(str(item) + '\n')

if __name__ == '__main__':
    main()
