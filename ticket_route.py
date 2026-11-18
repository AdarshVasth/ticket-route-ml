"""Train and explain a small local support ticket classifier."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

DATA = Path(__file__).resolve().parent / "data"
REVIEW_THRESHOLD = 0.55
MAX_LENGTH = 2_000


@dataclass(frozen=True)
class Route:
    queue: str
    suggested_queue: str
    scores: dict[str, float]
    supporting_terms: list[str]
    reason: str


def load_examples(path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return [row["ticket"] for row in rows], [row["queue"] for row in rows]


def train(path=DATA / "train.csv"):
    texts, queues = load_examples(path)
    model = Pipeline([
        ("terms", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        # A larger C lets this tiny demo separate strongly worded examples.
        ("router", LogisticRegression(C=12, max_iter=500)),
    ])
    return model.fit(texts, queues)


def route(model, ticket, threshold=REVIEW_THRESHOLD):
    ticket = ticket.strip()
    if not 0 < len(ticket) <= MAX_LENGTH:
        raise ValueError(f"Enter a ticket between 1 and {MAX_LENGTH} characters.")
    if not 0 <= threshold <= 1:
        raise ValueError("The review threshold must be between 0 and 1.")

    vectorizer = model.named_steps["terms"]
    classifier = model.named_steps["router"]
    features = vectorizer.transform([ticket])
    probabilities = classifier.predict_proba(features)[0]
    winner = int(probabilities.argmax())
    scores = dict(zip(classifier.classes_.tolist(), probabilities.tolist()))

    # Multiclass coefficients contribute to the suggested category's raw score.
    names = vectorizer.get_feature_names_out()
    row = features.tocoo()
    contributions = [
        (float(value * classifier.coef_[winner, column]), names[column])
        for column, value in zip(row.col, row.data)
    ]
    terms = [str(term) for weight, term in sorted(contributions, reverse=True) if weight > 0][:5]
    suggestion = str(classifier.classes_[winner])
    if features.nnz == 0:
        queue, reason = "human_review", "No terms matched the training vocabulary."
    elif probabilities[winner] < threshold:
        queue, reason = "human_review", "The top model score fell below the review threshold."
    else:
        queue, reason = suggestion, "The top model score met the review threshold."
    return Route(queue, suggestion, scores, terms, reason)


def evaluate(model, path=DATA / "holdout.csv"):
    texts, expected = load_examples(path)
    predictions = model.predict(texts)
    routes = [route(model, ticket).queue for ticket in texts]
    labels = model.classes_.tolist()
    print(classification_report(expected, predictions, labels=labels, zero_division=0))
    print("Confusion matrix (rows are true labels, columns are predictions)")
    print(", ".join(labels))
    print(confusion_matrix(expected, predictions, labels=labels))
    auto_routed = [(actual, queue) for actual, queue in zip(expected, routes) if queue != "human_review"]
    correct = sum(actual == queue for actual, queue in auto_routed)
    print(f"Auto-routed: {len(auto_routed)}/{len(routes)}. Correct among auto-routed: {correct}/{len(auto_routed)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("route", help="Route one ticket and show its evidence")
    predict.add_argument("ticket")
    predict.add_argument("--threshold", type=float, default=REVIEW_THRESHOLD)
    commands.add_parser("evaluate", help="Evaluate on the separate synthetic holdout")
    args = parser.parse_args()
    model = train()
    if args.command == "evaluate":
        evaluate(model)
        return
    try:
        result = route(model, args.ticket, args.threshold)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
