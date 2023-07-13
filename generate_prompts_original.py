# generate_prompts_original.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Generates 19 original prompts per movie.
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
    
    # Creates unenriched prompts.
    prompts_df['0a'] = df['title'].apply(lambda x: f'{x} is a movie of the genre [MASK].')
    prompts_df['0c'] = df['title'].apply(lambda x: f'{x} is a movie of the genre <mask>.')
    # prompts_df['0e'] = df['title'].apply(lambda x: f'{x} is a movie of the genre <s>.')
    # Creates original prompts.
    prompts_df['1a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, of the genre [MASK].', axis=1)
    prompts_df['2a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, of the genre [MASK].', axis=1)
    prompts_df['3a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, of the genre [MASK].', axis=1)
    prompts_df['4a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, of the genre [MASK].', axis=1)
    prompts_df['5a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, of the genre [MASK].', axis=1)
    prompts_df['6a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, of the genre [MASK].', axis=1)
    prompts_df['7a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, of the genre [MASK].', axis=1)
    prompts_df['8a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, distributed by {x.distributor}, of the genre [MASK].', axis=1)
    prompts_df['9a'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, distributed by {x.distributor}, originating from {x.country}, of the genre [MASK].', axis=1)

    prompts_df['1b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and is of the genre [MASK].', axis=1)
    prompts_df['2b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and is of the genre [MASK].', axis=1)
    prompts_df['3b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and is of the genre [MASK].', axis=1)
    prompts_df['4b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and is of the genre [MASK].', axis=1)
    prompts_df['5b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and is of the genre [MASK].', axis=1)
    prompts_df['6b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and is of the genre [MASK].', axis=1)
    prompts_df['7b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and is of the genre [MASK].', axis=1)
    prompts_df['8b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and distributed by {x.distributor} and is of the genre [MASK].', axis=1)
    prompts_df['9b'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and distributed by {x.distributor} and originating from {x.country} and is of the genre [MASK].', axis=1)

    prompts_df['1c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, of the genre <mask>.', axis=1)
    prompts_df['2c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, of the genre <mask>.', axis=1)
    prompts_df['3c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, of the genre <mask>.', axis=1)
    prompts_df['4c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, of the genre <mask>.', axis=1)
    prompts_df['5c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, of the genre <mask>.', axis=1)
    prompts_df['6c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, of the genre <mask>.', axis=1)
    prompts_df['7c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, of the genre <mask>.', axis=1)
    prompts_df['8c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, distributed by {x.distributor}, of the genre <mask>.', axis=1)
    prompts_df['9c'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, distributed by {x.distributor}, originating from {x.country}, of the genre <mask>.', axis=1)

    prompts_df['1d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and is of the genre <mask>.', axis=1)
    prompts_df['2d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and is of the genre <mask>.', axis=1)
    prompts_df['3d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and is of the genre <mask>.', axis=1)
    prompts_df['4d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and is of the genre <mask>.', axis=1)
    prompts_df['5d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and is of the genre <mask>.', axis=1)
    prompts_df['6d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and is of the genre <mask>.', axis=1)
    prompts_df['7d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and is of the genre <mask>.', axis=1)
    prompts_df['8d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and distributed by {x.distributor} and is of the genre <mask>.', axis=1)
    prompts_df['9d'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and distributed by {x.distributor} and originating from {x.country} and is of the genre <mask>.', axis=1)

    # prompts_df['1e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, of the genre <s>.', axis=1)
    # prompts_df['2e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, of the genre <s>.', axis=1)
    # prompts_df['3e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, of the genre <s>.', axis=1)
    # prompts_df['4e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, of the genre <s>.', axis=1)
    # prompts_df['5e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, of the genre <s>.', axis=1)
    # prompts_df['6e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, of the genre <s>.', axis=1)
    # prompts_df['7e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, of the genre <s>.', axis=1)
    # prompts_df['8e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, distributed by {x.distributor}, of the genre <s>.', axis=1)
    # prompts_df['9e'] = df.apply(lambda x: f'{x.title} is a movie, starring {x.cast}, directed by {x.director}, produced by {x.producer}, screenwriter {x.screenwriter}, music by {x.composer}, edited by {x.editor}, released {x.year}, distributed by {x.distributor}, originating from {x.country}, of the genre <s>.', axis=1)

    # prompts_df['1f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and is of the genre <s>.', axis=1)
    # prompts_df['2f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and is of the genre <s>.', axis=1)
    # prompts_df['3f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and is of the genre <s>.', axis=1)
    # prompts_df['4f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and is of the genre <s>.', axis=1)
    # prompts_df['5f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and is of the genre <s>.', axis=1)
    # prompts_df['6f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and is of the genre <s>.', axis=1)
    # prompts_df['7f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and is of the genre <s>.', axis=1)
    # prompts_df['8f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and distributed by {x.distributor} and is of the genre <s>.', axis=1)
    # prompts_df['9f'] = df.apply(lambda x: f'{x.title} is a movie starring {x.cast} and directed by {x.director} and produced by {x.producer} and screenwriter {x.screenwriter} and music by {x.composer} and edited by {x.editor} and released {x.year} and distributed by {x.distributor} and originating from {x.country} and is of the genre <s>.', axis=1)


    # Saves original prompts to CSV file.
    prompts_df.to_csv(os.path.join("Data", 'movies_prompts_original.csv'), index=False)

if __name__ == '__main__':
    main()
    