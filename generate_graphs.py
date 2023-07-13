import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

if not os.path.exists("Graphs"):
    os.makedirs("Graphs")
    
data = pd.read_csv(os.path.join("Results", 'recall_stats.csv'))

def process_data():
    data.set_index('llm', inplace=True)

    recall_1_cols = [col for col in data.columns if 'recall@1_' in col and 'recall@10' not in col]
    recall_5_cols = [col for col in data.columns if 'recall@5_' in col]
    recall_10_cols = [col for col in data.columns if 'recall@10_' in col]

    recall_1 = data[recall_1_cols]
    recall_5 = data[recall_5_cols]
    recall_10 = data[recall_10_cols]

    # Remove the 'recall@_' part from the column names
    recall_1.columns = recall_1.columns.str.replace('recall@1_', '')
    recall_5.columns = recall_5.columns.str.replace('recall@5_', '')
    recall_10.columns = recall_10.columns.str.replace('recall@10_', '')

    return recall_1, recall_5, recall_10

def heatmap(data, title):
    sns.set(rc={'figure.figsize':(11.0, 8.0)})  # Set figure size
    sns.heatmap(data, cmap="YlGnBu", annot=False, cbar=True)  # Turn off automatic annotations
    plt.title(title)
    plt.xlabel('Prompt Style')
    plt.ylabel('Large Language Model')

    threshold = data.max().max()/2  # Set a threshold at 50% of the max value

    # Add annotations manually
    for i in range(data.shape[0]):  # for each row
        for j in range(data.shape[1]):  # for each column
            if data.iloc[i, j] == data.iloc[i].max():  # if cell value is the maximum of the row
                # annotate with bold text
                plt.text(j+0.5, i+0.5, f'{data.iloc[i, j]:.3f}', 
                         horizontalalignment='center', 
                         verticalalignment='center', 
                         fontweight='bold',
                         color='red')
            else:
                # regular annotation, color based on cell value
                text_color = 'black' if data.iloc[i, j] < threshold else 'white'
                plt.text(j+0.5, i+0.5, f'{data.iloc[i, j]:.3f}', 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         color=text_color) 
    plt.show()

def bubble():
    params = {
        "bert-base-uncased": 109_482_240,
        "xlm-roberta-large": 355_359_744,
        "bart-large": 406_290_432,
        "deberta-large": 400_522_752,
        "xlnet-large-cased": 360_268_800,
    }

    # create a new column 'params' based on the 'LLM' column
    data['params'] = data['LLM'].map(params)

    # normalize parameters for better visualization
    data['params'] = data['params'] / 1e6

    # average the recalls and parameters for each prompt style
    averaged_data = data.groupby('Prompt_Style').mean().reset_index()

    # define a color palette
    palette = sns.color_palette('hsv', 10)
    light_palette = [sns.light_palette(color, n_colors=3)[1] for color in palette]

    # separate the numeric and alpha parts of the prompt_style to have 10 groups
    averaged_data['prompt_num'] = averaged_data['Prompt_Style'].apply(lambda x: x[:-1] if x[-1].isalpha() else x)
    averaged_data['prompt_letter'] = averaged_data['Prompt_Style'].apply(lambda x: x[-1] if x[-1].isalpha() else '')

    # create a color dictionary mapping prompt styles to colors
    color_dict = {str(i): palette[i] for i in range(10)}
    color_dict.update({str(i) + 'a': palette[i] for i in range(10)})
    color_dict.update({str(i) + 'b': light_palette[i] for i in range(10)})

    # plotting the data
    fig, ax = plt.subplots(figsize=(10, 6))

    for index, row in averaged_data.iterrows():
        color = color_dict[row['prompt_num'] + row['prompt_letter']]
        ax.scatter(row['Average_recall@1'], row['Average_recall@5'], s=row['params'], label=row['Prompt_Style'], color=color, alpha=0.6, edgecolors='w')

    ax.set_xlabel('Average_recall@1')
    ax.set_ylabel('Average_recall@5')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def downloads_bar_chart():
    download_counts = {
        "BERT": 53_800_000,
        "RoBERTa large": 30_000_000,
        "DeBERTa large": 20_000,
        "XLNet large": 45000,
        "BART large": 39000
    }

    # Create a bar chart
    plt.bar(range(len(download_counts)), list(download_counts.values()), align='center')
    plt.xticks(range(len(download_counts)), list(download_counts.keys()))
    plt.xlabel('LLMs')
    plt.ylabel('Number of Downloads')
    plt.title('Number of Downloads for Each LLM from Hugging Face')
    plt.show()





def main():
    recall_1, recall_5, recall_10 = process_data()
    heatmap(recall_1, 'Recall@1')
    heatmap(recall_5, 'Recall@5')
    heatmap(recall_10, 'Recall@10')

if __name__ == "__main__":
    main()
