# clean_movies_kg.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Cleans a dataset of movies knowledge graph properties.
# 

# Imports.
import pandas as pd
import re
import os

def remove_special_char_words(s):
    """Removes words containing special characters from a string.

    Checks if the input is a string, splits the string into words,
    removes any words that contain special characters, and then 
    joins the words back together.

    Args:
        s (str): The input string.

    Returns:
        str: The input string with words containing special characters removed.
        Returns the original input if it is not a string.
    """
    if isinstance(s, str):
        words = s.split()
        words = [word for word in words if re.match("^[a-zA-Z0-9_.\-|']*$", word)]
        s = ' '.join(words)
    return s

def main():
    df = pd.read_csv(os.path.join("Data", 'movies_kg_full.csv'))
    
    # Get the names of columns with string data type
    string_columns = df.select_dtypes(include=[object]).columns.tolist()
    
    # Apply the remove_special_char_words function to the entire DataFrame
    df = df.applymap(remove_special_char_words)

    # Remove rows containing only whitespace in string columns
    df = df[~df[string_columns].apply(lambda series: series.str.contains(r'^ *$', na=False)).any(axis=1)]

    # Remove rows containing 'Q' followed by a digit in string columns
    df = df[~df[string_columns].apply(lambda series: series.str.contains(r'Q\d+', na=False)).any(axis=1)]

    # Drop rows with NaN values and reset the index
    df = df.dropna(how='any').reset_index(drop=True)
    
    # Calculates total times each genre is mentioned.
    genres_df = df['genres'].str.get_dummies('|')
    genre_counts = genres_df.sum()
    print("Genre counts across whole dataset:")
    print(genre_counts)
    
    # Write the cleaned DataFrame to a csv file
    df.to_csv(os.path.join("Data", 'movies_kg_cleaned.csv'), index=False)

if __name__ == '__main__':
    main()
