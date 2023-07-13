PYTHON FILES

fetch_movies_kg: Retrieves KG properties Section 4.2 (Knowledge Graph Querying)

clean_movies_kg: Cleans KG dataset Section 4.2 (Knowledge Graph Querying)

generate_prompts_original: Generates original prompts from Brate et al. (Section 4.3 Prompt Engineering)

generate_prompts_custom: Generates custom prompts constructed in this paper (Section 4.3 Prompt Engineering)

predict_movies_original: Probes LLMs with original prompts (Section 4.4 Large Language Model Probing)

predict_movies_custom: Probes LLMs with custom prompts (Section 4.4 Large Language Model Probing)

statistical_evaluation: Statistically analyses the results (Section 5 Results)

t_tests: Runs statistical significance tests (Section 5 Results)


DIRECTORIES

Data: Stores raw movies dataset, cleaned movies dataset, movies knowledge graph information, cleaned movies knowledge graph information, original and custom prompts, and the Wikidata query error file.

Graphs: Stores graphs used in report.

Predictions: Stores large language model movie genre predictions for each model.

Recall: Stores recall of model predictions.

Results: Stores results of statistical analysis and t-tests.