# predict_movies_original.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Probes a range of LLMs with the constructed original prompts.
# 

# Imports.
import pandas as pd
import os
import time
import torch
from transformers import (BertTokenizer, BertForMaskedLM, 
                          RobertaTokenizer, RobertaForMaskedLM,
                          DebertaTokenizer, DebertaForMaskedLM,
                          BartTokenizer, BartForConditionalGeneration,
                          XLNetTokenizer, XLNetLMHeadModel,
                          ElectraTokenizer, ElectraForMaskedLM,
                          AlbertTokenizer, AlbertForMaskedLM)

def main():
    if not os.path.exists("Predictions"):
        os.makedirs("Predictions")
    # Reads the movie prompts dataset.
    df = pd.read_csv(os.path.join("Data", 'movies_prompts_original.csv')).head(10)
    # Extracts all unique genres from the dataset.
    unique_genres = pd.unique([item for sublist in df['genres'].str.split('|') for item in sublist])
    print("All unique genres:", unique_genres)
    # Sets the computation device based on availability.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Defines the models to be used for predictions.
    models = [('bert-base-uncased', BertTokenizer, BertForMaskedLM, '0a', 'a', 'b'), 
              ('roberta-large', RobertaTokenizer, RobertaForMaskedLM, '0c', 'c', 'd'),
              ('microsoft/deberta-large', DebertaTokenizer, DebertaForMaskedLM, '0a', 'a', 'b'),
              ('facebook/bart-large', BartTokenizer, BartForConditionalGeneration, '0c', 'c', 'd'),
              ('xlnet-large-cased', XLNetTokenizer, XLNetLMHeadModel, '0c', 'c', 'd'),
              ('google/electra-large-discriminator', ElectraTokenizer, ElectraForMaskedLM, '0a', 'a', 'b'),
              ('albert-large-v2', AlbertTokenizer, AlbertForMaskedLM, '0a', 'a', 'b')]
    
    # Iterate over each model for prediction.
    for model_name, Tokenizer, Model, column_0, column_1, column_2 in models:
        start_time = time.time()
        print("Current LLM:", model_name)
        df_model = df.copy()  
        # Loads model and tokenizer.
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = Model.from_pretrained(model_name)
        model.to(device)
        model.eval()
        # Converts genre names to corresponding token IDs.
        genre_ids = []
        genres = []
        for genre in unique_genres:
            genre_ids.append(tokenizer.convert_tokens_to_ids(genre))
            genres.append(genre)
            
        # Defines the list of prompt style and results columns.
        prompt_columns = [f'{i}{c}' for i in range(1, 10) for c in [column_1, column_2]]
        result_columns = ["title", "genres"]
        # Iterates over each prompt style.
        for prompt_column in [column_0] + prompt_columns:
            print("Current prompt style:", prompt_column)
            # Standardizes the results columns.
            result_column_base = prompt_column.replace('0a', '0').replace('0c', '0').replace('c', 'a').replace('d', 'b')
            result_columns.extend([f'{result_column_base}'])
            # Generates predictions for each prompt in the current column.
            for i, prompt in enumerate(df_model[prompt_column]):
                try:
                    # Prepares input tensor for model.
                    tokens = tokenizer.encode(prompt, add_special_tokens=True)
                    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
                    # Generates and process model predictions.
                    with torch.no_grad():
                        predictions = model(input_ids).logits[0, tokens.index(tokenizer.mask_token_id)]
                    predicted_genres = [genres[id] for id in torch.topk(predictions[genre_ids], 10).indices]
                        
                    # Stores top 10 genre predictions in the DataFrame.
                    df_model.at[i, f'{result_column_base}'] = "|".join(predicted_genres)

                except Exception as e:
                    print("Exception: " + str(e))

        # Saves results to CSV file.
        df_model[result_columns].to_csv(os.path.join("Predictions", f"{model_name.split('/')[-1]}_original.csv"), index=False)
        time_taken = time.time() - start_time
        print("Time taken:", time_taken)
        # Saves the time taken for the current LLM in a separate text file.
        with open(os.path.join("Predictions", f"{model_name.split('/')[-1]}_time_taken_original.txt"), "w") as file:
            file.write(f"Time taken: {time_taken}")

if __name__ == '__main__':
    main()
    