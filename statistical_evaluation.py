# statistical_evaluation.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Statistically evaluates the movie prediction results.
# 

# Imports.
import pandas as pd
import os
from collections import Counter

def calculate_recall(model_name, df):
    """Calculates and store recall at positions 1, 5, and 10 for each result
       column.

    Args:
    model_name: str
        Name of the model.
    df: DataFrame
        Input DataFrame.

    Returns:
        DataFrame: DataFrame with recall values for each result column.
    """
    # Defines result columns.
    result_columns = ['0'] + [str(i) + 'a' for i in range(1, 19)] + [str(i) + 'b' for i in range(1, 10)]
    for result_column in result_columns:  # For each results column.
        for i in range(len(df)):
            # Gets genres.
            predicted_genres = df.at[i, result_column].split('|')
            actual_genres = df.at[i, "genres"].split('|')
            # Calculates recall@1, recall@5 and recall@10.
            df.at[i, f'recall@1_{result_column}'] = int(predicted_genres[0] in actual_genres)
            df.at[i, f'recall@5_{result_column}'] = len([value for value in actual_genres if value in predicted_genres[:5]]) / len(actual_genres)
            df.at[i, f'recall@10_{result_column}'] = len([value for value in actual_genres if value in predicted_genres]) / len(actual_genres)

    # Exports DataFrame.
    df_export = df[[column for column in df.columns if column == 'title' or column.startswith('recall@')]]
    df_export.to_csv(os.path.join("Recall", f"{model_name.split('/')[-1]}.csv"), index=False)

    # Calculate average recall per movie
    movie_recall = df.set_index('title')[[column for column in df.columns if column.startswith('recall@1_')]].mean(axis=1)
    return df, movie_recall

def calculate_stats(df):
    """Calculates statistics and average genre counts across all styles.

    Args:
    df: DataFrame
        Input DataFrame.

    Returns:
    dict, dict: Dictionaries containing stats and genre counts.
    """
    # Defines prompt styles and recall_columns.
    styles = ['0'] + [f'{i}{suffix}' for i in range(1, 10) for suffix in ['a', 'b']] + [f'{i}a' for i in range(10, 19)]
    recall_columns = [f'recall@{i}_0' for i in [1,5,10]] + [f'recall@{i}_{j}{suffix}' for i in [1,5,10] for j in range(1, 10) for suffix in ['a', 'b']] + [f'recall@{i}_{j}a' for i in [1,5,10] for j in range(10, 19)]
    
    genre_counts_styles = {1: Counter(), 5: Counter(), 10: Counter()}

    # Calculates genre counts across all styles.
    for style in styles:  # For each prompt style.
        for recall in [1, 5, 10]:  # For each recall.
            for row in df[style]:
                genres = row.split('|')[:recall]
                for genre in genres:
                    genre_counts_styles[recall][genre] += 1

    # Normalize the genre counts so that the probabilities add up to 1.
    for recall in [1, 5, 10]:
        total_counts = sum(genre_counts_styles[recall].values())
        genre_counts_styles[recall] = {genre: count / total_counts for genre, count in genre_counts_styles[recall].items()}
            
    # Calculates average accuracy per recall.
    avg_recall = {column: df[column].mean() for column in recall_columns}
    avg_recall_1 = sum([v for k, v in avg_recall.items() if "recall@1" in k]) / len([k for k in avg_recall.keys() if "recall@1" in k])
    avg_recall_5 = sum([v for k, v in avg_recall.items() if "recall@5" in k]) / len([k for k in avg_recall.keys() if "recall@5" in k])
    avg_recall_10 = sum([v for k, v in avg_recall.items() if "recall@10" in k]) / len([k for k in avg_recall.keys() if "recall@10" in k])
    
    # Returns these statistics.
    stats = {**avg_recall, 'average_recall@1': avg_recall_1, 'average_recall@5': avg_recall_5, 'average_recall@10': avg_recall_10}
    return stats, genre_counts_styles

def calculate_genre_accuracy(df, recall_at):
    """Calculates accuracy for each genre.

    Args:
    df: DataFrame
        Input DataFrame.
    recall_at: int
        Level of recall to calculate accuracy at (1, 5, or 10).

    Returns:
    dict: Dictionary containing genre accuracy.
    """
    # Defines result columns.
    result_columns = ['0'] + [str(i) + 'a' for i in range(1, 19)] + [str(i) + 'b' for i in range(1, 10)]
    genre_hits = Counter()
    genre_counts = Counter()

    for result_column in result_columns:  # For each results column.
        for i in range(len(df)):
            # Gets genres.
            predicted_genres = df.at[i, result_column].split('|')[:recall_at]
            actual_genres = df.at[i, "genres"].split('|')
            # Calculates genre hits and counts.
            for genre in predicted_genres:
                genre_counts[genre] += 1
                if genre in actual_genres:
                    genre_hits[genre] += 1

    # Calculates genre accuracy.
    genre_accuracy = {genre: hits / genre_counts[genre] for genre, hits in genre_hits.items()}

    return genre_accuracy

def main():
    # Creates necessary directories if they do not exist.
    if not os.path.exists("Recall"):
        os.makedirs("Recall")
    if not os.path.exists("Results"):
        os.makedirs("Results")
    
    llm_names = ['bert-base-uncased', 'roberta-large', 'microsoft/deberta-large', 'facebook/bart-large', 'xlnet-large-cased', 'google/electra-large-discriminator', 'albert-large-v2']
    all_stats = []
    all_genre_counts = {1: [], 5: [], 10: []}
    rename_dict = {
        'bert-base-uncased': 'BERT', 
        'roberta-large': 'RoBERTa Large', 
        'microsoft/deberta-large': 'DeBERTa Large', 
        'facebook/bart-large': 'BART Large', 
        'xlnet-large-cased': 'XLNet Large', 
        'google/electra-large-discriminator': 'ELECTRA Large', 
        'albert-large-v2': 'ALBERT Large v2'
    }
    
    all_genre_accuracy = {1: [], 5: [], 10: []}
    movie_recalls = []
    
    for llm in llm_names:  # For each LLM.
        # Reads CSV files and merge original and custom prompt style DataFrames.
        original = pd.read_csv(os.path.join("Predictions", f"{llm.split('/')[-1]}_original.csv"))
        custom = pd.read_csv(os.path.join("Predictions", f"{llm.split('/')[-1]}_custom.csv"))
        # Drops unnecessary columns.
        custom.drop('genres', axis=1, inplace=True)
        custom.drop('title', axis=1, inplace=True)
        # Joins all predictions together.
        all_predictions_df = original.join(custom)
        # Calculates recall averages and other statistics.
        recall_df, movie_recall = calculate_recall(llm, all_predictions_df)
        movie_recalls.append(movie_recall)
        stats, genre_counts_styles = calculate_stats(recall_df)
        # Renames the LLMs.
        llm = rename_dict.get(llm, llm)
        # Adds current LLM's statistics to "all_stats".
        stats_df = pd.DataFrame(stats, index=[llm])
        all_stats.append(stats_df)
        # Calculates and stores genre counts.
        for recall in [1, 5, 10]:
            genre_counts_df = pd.DataFrame.from_dict(genre_counts_styles[recall], orient='index')
            genre_counts_df.columns = ['Count']
            genre_counts_df = genre_counts_df.fillna(0)
            genre_counts_df.columns = pd.MultiIndex.from_product([[llm], genre_counts_df.columns])
            all_genre_counts[recall].append(genre_counts_df)
            
            genre_accuracy = calculate_genre_accuracy(recall_df, recall)
            # Renames the LLMs.
            llm = rename_dict.get(llm, llm)
            # Adds current LLM's genre accuracy to "all_genre_accuracy".
            genre_accuracy_df = pd.DataFrame(genre_accuracy, index=[llm])
            all_genre_accuracy[recall].append(genre_accuracy_df)

    # Saves all statistics to CSV file.
    all_stats_df = pd.concat(all_stats, axis=0).reset_index()
    all_stats_df.rename(columns={'index':'llm'}, inplace=True)
    all_stats_df.to_csv(os.path.join("Results", 'recall_stats.csv'), index=False)
    
    # Combine all movie recall Series into a DataFrame and save it to a CSV file
    movie_recalls_df = pd.concat(movie_recalls, axis=1)
    movie_recalls_df.columns = llm_names

    # Add average recall per movie column
    movie_recalls_df['Average'] = movie_recalls_df.mean(axis=1)

    movie_recalls_df.to_csv(os.path.join("Results", 'movie_recalls.csv'))
    
    # Saves all genre counts to CSV file.
    for recall in [1, 5, 10]:  # For each recall.
        all_genre_counts_df = pd.concat(all_genre_counts[recall], axis=1)
        all_genre_counts_df.to_csv(os.path.join("Results", f'genre_counts_{recall}.csv'))
        
        all_genre_accuracy_df = pd.concat(all_genre_accuracy[recall])

        # Add an average row
        avg_row = all_genre_accuracy_df.mean()
        all_genre_accuracy_df.loc['Average'] = avg_row

        all_genre_accuracy_df.T.to_csv(os.path.join("Results", f'genre_accuracy_{recall}.csv'))

if __name__ == '__main__':
    main()

    