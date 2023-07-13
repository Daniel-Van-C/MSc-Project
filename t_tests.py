# t_tests.py
# Daniel Van Cuylenburg (k19012373)
# 01/07/2023
# 
# Runs statistical significance tests for different recalls and prompt styles
# for each movie.
# 

# Imports.
import pandas as pd
from scipy import stats
import os

def main():
    llm_names = ['bert-base-uncased', 'roberta-large', 'deberta-large', 'bart-large', 'xlnet-large-cased', 'electra-large-discriminator', 'albert-large-v2']
    results = pd.DataFrame(columns=["LLM", "Recall", "Prompt", "Mean Difference", "Test Statistic", "P-Value"])

    for llm in llm_names:  # For each LLM.
        # Reads recall results.
        df = pd.read_csv(os.path.join("Recall", llm + '.csv'))

        # Iterates over different recalls.
        for recall in ['recall@1', 'recall@5', 'recall@10']:
            # Selects unenriched prompt column.
            unenriched = f'{recall}_0'
            
            for prompt in range(1, 19):  # Iterates over enriched prompt styles.
                for style in ['a', 'b']:
                    # Skips 'b' styles for custom prompt styles.
                    if prompt in ([0] + list(range(10, 19))) and style == 'b':
                        continue
                    # Selects current enriched prompt column.
                    enriched = f'{recall}_{prompt}{style}'
                    # Performs paired t-test.
                    t_stat, p_val = stats.ttest_rel(df[unenriched], df[enriched], alternative='less')
                    mean_diff = df[enriched].mean() - df[unenriched].mean()
                    # Stores the result.
                    result = pd.Series([llm, recall, f'{prompt}{style}', mean_diff, t_stat, p_val])
                    results = pd.concat([results, result], axis=1)

    # Transposes and clean up the results DataFrame.
    results = results.T
    results.columns = ["LLM", "Recall", "Prompt", "Mean Difference", "Test Statistic", "P-Value"]
    results = results[results["LLM"].notna()]
    results.reset_index(drop=True, inplace=True)
    # Saves the results in a CSV file.
    results.to_csv(os.path.join("Results", 't_tests.csv'), index=False)

if __name__ == '__main__':
    main()
