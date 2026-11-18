# Ticket Route ML

Route English customer support tickets to account access, billing, delivery, or product bug. The local command-line demo prints a suggested queue, category scores, and words or phrases that contributed to the suggestion. When evidence is sparse or the top score is low, the ticket is marked for human review.

## Run locally

Use Python 3.11 or later. The project was tested with scikit-learn 1.8.0. After installing dependencies, the demo runs offline with no account or API key.

```sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ticket_route.py route "My package is missing but the courier marked it delivered"
python ticket_route.py route "quasar intergalactic nebula"
python ticket_route.py evaluate
python -m unittest discover -s tests -v
```

Use `--threshold 0.7` after the ticket text to require a stronger score for automatic routing. `human_review` is a routing decision in the JSON output. A real help desk would still need to store and assign those tickets.

## How it works

1. Each invocation trains a [scikit-learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) on 48 synthetic tickets in `data/train.csv`.
2. [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) turns individual words and adjacent word pairs into numeric features. [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) learns one set of feature weights for each category.
3. The model estimates a score for each queue. It sends tickets to review if the vocabulary is empty or the highest score falls below 0.55. Explanatory terms are the strongest positive contributions to the suggested queue’s raw score.
4. The evaluate command reports raw classification metrics and review coverage on 12 separate synthetic examples. With the default threshold, the model suggested the correct category on 12 of 12 examples and automatically routed 10 of them. All 10 automatic routes were correct.

Both CSVs were written for this demo. The sample data are available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## Limitations and learning guide

The synthetic holdout is small and uses vocabulary similar to the training set. These results only describe those examples. Real tickets can be multilingual, ambiguous, or cover multiple issues. Scores are uncalibrated and the threshold is a demo choice. The CLI prints only the scores and matched terms, but command-line arguments can appear in shell history, so use fictional examples.

- Run the same ticket at two thresholds and explain how the review cutoff changes coverage.
- Try a paraphrase and compare its score and supporting terms with the original.
- Add unrelated and multi-issue examples to your own holdout. Compare the error rate among automatic routes with the fraction sent to review.
