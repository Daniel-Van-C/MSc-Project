# generate_prompts_custom.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Generates 9 custom prompts per movie.
# 

# Imports.
import pandas as pd
import os

def main():
    # Read cleaned movie data.
    df = pd.read_csv(os.path.join("Data", 'movies_kg_cleaned.csv'))
    prompts_df = df[['movieId', 'title', 'genres']].copy()
    # Lowercase genres for uniformity
    prompts_df['genres'] = prompts_df['genres'].str.lower()
    # Creates custom prompts.
    prompts_df['10a'] = df.apply(lambda x: f'The movie {x.title} was released in {x.year}. It starred {x.cast} and {x.director} directed it. Can you guess the genre? [MASK]', axis=1)
    prompts_df['11a'] = df.apply(lambda x: f'The movie {x.title} was released in {x.year}. It starred {x.cast} and {x.director} directed it. The script was written by {x.screenwriter} and the music was composed by {x.composer}. Can you guess the genre? [MASK]', axis=1)
    prompts_df['12a'] = df.apply(lambda x: f'Can you tell me the genre of the movie {x.title}, released in {x.year} starring {x.cast}? [MASK]', axis=1)
    prompts_df['13a'] = df.apply(lambda x: f'Can you tell me the genre of the movie {x.title}, released in {x.year} directed by {x.director}? [MASK]', axis=1)
    prompts_df['14a'] = df.apply(lambda x: f'The movie {x.title} is a film starring {x.cast}, directed by {x.director}, produced by {x.producer}, with the screenplay by {x.screenwriter}. It has music by {x.composer}, editing by {x.editor}, was released in {x.year}, and distributed by {x.distributor}. It originates from {x.country}. What is its genre? [MASK]', axis=1)
    prompts_df['15a'] = df.apply(lambda x: f'From the mind of {x.director} and brought to life by {x.cast}, {x.title} is a noteworthy addition to the [MASK] genre.', axis=1)
    prompts_df['16a'] = df.apply(lambda x: f'With {x.title}, {x.director} brings a new twist to the [MASK] genre, featuring powerful performances by {x.cast}.', axis=1)
    prompts_df['17a'] = df.apply(lambda x: f'The [MASK] genre was beautifully represented in {x.country} through the movie {x.title}, featuring the unique performance of {x.cast}.', axis=1)
    prompts_df['18a'] = df.apply(lambda x: f'A film from {x.country}, {x.title} features {x.cast} and falls into the [MASK] genre under the direction of {x.director}.', axis=1)

    prompts_df['10c'] = df.apply(lambda x: f'The movie {x.title} was released in {x.year}. It starred {x.cast} and {x.director} directed it. Can you guess the genre? <mask>', axis=1)
    prompts_df['11c'] = df.apply(lambda x: f'The movie {x.title} was released in {x.year}. It starred {x.cast} and {x.director} directed it. The script was written by {x.screenwriter} and the music was composed by {x.composer}. Can you guess the genre? <mask>', axis=1)
    prompts_df['12c'] = df.apply(lambda x: f'Can you tell me the genre of the movie {x.title}, released in {x.year} starring {x.cast}? <mask>', axis=1)
    prompts_df['13c'] = df.apply(lambda x: f'Can you tell me the genre of the movie {x.title}, released in {x.year} directed by {x.director}? <mask>', axis=1)
    prompts_df['14c'] = df.apply(lambda x: f'The movie {x.title} is a film starring {x.cast}, directed by {x.director}, produced by {x.producer}, with the screenplay by {x.screenwriter}. It has music by {x.composer}, editing by {x.editor}, was released in {x.year}, and distributed by {x.distributor}. It originates from {x.country}. What is its genre? <mask>', axis=1)
    prompts_df['15c'] = df.apply(lambda x: f'From the mind of {x.director} and brought to life by {x.cast}, {x.title} is a noteworthy addition to the <mask> genre.', axis=1)
    prompts_df['16c'] = df.apply(lambda x: f'With {x.title}, {x.director} brings a new twist to the <mask> genre, featuring powerful performances by {x.cast}.', axis=1)
    prompts_df['17c'] = df.apply(lambda x: f'The <mask> genre was beautifully represented in {x.country} through the movie {x.title}, featuring the unique performance of {x.cast}.', axis=1)
    prompts_df['18c'] = df.apply(lambda x: f'A film from {x.country}, {x.title} features {x.cast} and falls into the <mask> genre under the direction of {x.director}.', axis=1)
    
    # Saves custom prompts to CSV file.
    prompts_df.to_csv(os.path.join("Data", 'movies_prompts_custom.csv'), index=False)

if __name__ == '__main__':
    main()