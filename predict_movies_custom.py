# predict_movies_custom.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Probes a range of LLMs with the constructed custom prompts.
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
    # Reads the movie prompts dataset.
    df = pd.read_csv(os.path.join("Data", 'movies_prompts_custom.csv'))
    # Extracts all unique genres from the dataset.
    genres = pd.unique([item for sublist in df['genres'].str.split('|') for item in sublist])
    # Sets the computation device based on availability.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Defines the models to be used for predictions.
    models = [('bert-base-uncased', BertTokenizer, BertForMaskedLM, '10a'), 
              ('roberta-large', RobertaTokenizer, RobertaForMaskedLM, '10c'),
              ('microsoft/deberta-large', DebertaTokenizer, DebertaForMaskedLM, '10a'),
              ('facebook/bart-large', BartTokenizer, BartForConditionalGeneration, '10c')
              ('xlnet-large-cased', XLNetTokenizer, XLNetLMHeadModel, '10c'),
              ('google/electra-large-discriminator', ElectraTokenizer, ElectraForMaskedLM, '10a'),
              ('albert-large-v2', AlbertTokenizer, AlbertForMaskedLM, '10a')]

    # Iterate over each model for prediction.
    for model_name, Tokenizer, Model, column_id in models:
        start_time = time.time()
        print("Current LLM:", model_name)
        df_model = df.copy()
        # Loads model and tokenizer.
        tokenizer = Tokenizer.from_pretrained(model_name)
        model = Model.from_pretrained(model_name)
        model.to(device)
        model.eval()
        # Converts genre names to corresponding token IDs.
        genre_ids = [tokenizer.convert_tokens_to_ids(word) for word in genres]
        # Defines the list of prompt style and results columns..
        prompt_columns = [f'{i}{column_id[-1]}' for i in range(11, 19)]
        result_columns = ["title", "genres"] 
        # Iterates over each prompt style.
        for prompt_column in [column_id] + prompt_columns:
            print("Current prompt style:", prompt_column)
            # Standardizes the results columns.
            result_column_base = prompt_column.replace('c', 'a')
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
        df_model[result_columns].to_csv(os.path.join("Predictions", f"{model_name.split('/')[-1]}_custom.csv"), index=False)
        time_taken = time.time() - start_time
        print("Time taken:", time_taken)
        # Saves the time taken for the current LLM in a separate text file.
        with open(os.path.join("Predictions", f"{model_name.split('/')[-1]}_time_taken_custom.txt"), "w") as file:
            file.write(f"Time taken: {time_taken}")

if __name__ == '__main__':
    main()
    