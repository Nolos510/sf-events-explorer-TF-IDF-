"""
Model comparison: TF-IDF vs CountVectorizer for SF Events Explorer

Usage:
    python model_comparison_countvectorizer.py

This script:
- Loads events.csv
- Builds search_text (same as app.py)
- Trains two models:
    1) TF-IDF + cosine similarity
    2) CountVectorizer + cosine similarity
- Evaluates both using:
    - Precision@10
    - Mean Reciprocal Rank (MRR)
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load data & build search_text (same logic as in app.py)
def load_events():
    paths_to_try = [
        "events.csv",
        "./events.csv",
        "data/events.csv",
        "./data/events.csv",
    ]

    df = None
    for path in paths_to_try:
        try:
            df = pd.read_csv(path)
            print(f"Loaded data from: {path}")
            break
        except FileNotFoundError:
            continue

    if df is None:
        raise FileNotFoundError("Could not find events.csv in expected locations.")

    # Recreate search_text exactly as in app.py
    df["event_name"] = df["event_name"].fillna("")
    df["event_description"] = df["event_description"].fillna("")
    df["category"] = df["category"].fillna("")

    df["search_text"] = (
        df["event_name"] + " " +
        df["event_description"] + " " +
        df["category"]
    )

    return df


# Build both models (TF-IDF & CountVectorizer)
def build_models(df):
    corpus = df["search_text"].tolist()

    # Model A: TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Model B: CountVectorizer (bag-of-words)
    count_vectorizer = CountVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    count_matrix = count_vectorizer.fit_transform(corpus)

    return (tfidf_vectorizer, tfidf_matrix), (count_vectorizer, count_matrix)


#  search function for each model
def search_events(query, vectorizer, matrix, top_k=10):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]


# Evaluation metrics: Precision@K & MRR
def precision_at_k(relevant_mask, k=10):
    """
    relevant_mask: boolean array of length k (True if item at rank i is relevant)
    """
    k = min(k, len(relevant_mask))
    if k == 0:
        return 0.0
    return relevant_mask[:k].sum() / k


def reciprocal_rank(relevant_mask):
    """
    1 / rank of first relevant item (1-based).
    0 if no relevant items.
    """
    for i, rel in enumerate(relevant_mask):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


# Define test queries & automatic relevance rules
def build_test_queries():
    """
    Each test query has:
        - text: the search query
        - relevance_keywords: list of keywords that SHOULD appear
          in relevant events (in search_text)
    This is a heuristic, not perfect, but enough to compare models.
    """

    test_queries = [
        {
            "text": "fun activities for kids",
            "relevance_keywords": ["kid", "child", "children", "family", "toddler"],
        },
        {
            "text": "free art classes",
            "relevance_keywords": ["art", "drawing", "painting", "class"],
        },
        {
            "text": "sports & recreation",
            "relevance_keywords": ["sport", "soccer", "basketball", "recreation"],
        },
        {
            "text": "computer coding workshop",
            "relevance_keywords": ["coding", "computer", "programming", "tech"],
        },
        {
            "text": "music performance",
            "relevance_keywords": ["music", "concert", "performance"],
        },
        {
            "text": "job fair & career support",
            "relevance_keywords": ["career", "job", "employment", "resume"],
        },
    ]

    return test_queries


def is_relevant(row_text, relevance_keywords):
    text = row_text.lower()
    return any(kw in text for kw in relevance_keywords)


#  Run evaluation for both models
def evaluate_models(df, model_a, model_b, top_k=10):
    (tfidf_vectorizer, tfidf_matrix) = model_a
    (count_vectorizer, count_matrix) = model_b

    test_queries = build_test_queries()

    results = []
    for q in test_queries:
        query_text = q["text"]
        keywords = q["relevance_keywords"]

        # Model A: TF-IDF
        idx_a, _ = search_events(query_text, tfidf_vectorizer, tfidf_matrix, top_k=top_k)
        # Build relevance mask for top_k results
        relevant_mask_a = []
        for i in idx_a:
            row_text = df.iloc[i]["search_text"]
            relevant_mask_a.append(is_relevant(row_text, keywords))
        relevant_mask_a = np.array(relevant_mask_a)

        p_at_10_a = precision_at_k(relevant_mask_a, k=top_k)
        rr_a = reciprocal_rank(relevant_mask_a)

        # Model B: CountVectorizer
        idx_b, _ = search_events(query_text, count_vectorizer, count_matrix, top_k=top_k)
        relevant_mask_b = []
        for i in idx_b:
            row_text = df.iloc[i]["search_text"]
            relevant_mask_b.append(is_relevant(row_text, keywords))
        relevant_mask_b = np.array(relevant_mask_b)

        p_at_10_b = precision_at_k(relevant_mask_b, k=top_k)
        rr_b = reciprocal_rank(relevant_mask_b)

        results.append(
            {
                "query": query_text,
                "P@10_TFIDF": p_at_10_a,
                "MRR_TFIDF": rr_a,
                "P@10_Count": p_at_10_b,
                "MRR_Count": rr_b,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


# Main
def main():
    df = load_events()
    print(f"Total events: {len(df)}")

    model_a, model_b = build_models(df)
    print("Models trained: TF-IDF & CountVectorizer")

    results_df = evaluate_models(df, model_a, model_b, top_k=10)
    print("\nPer-query results:")
    print(results_df.to_string(index=False))

    print("\nMean metrics across all test queries:")
    print(
        results_df[
            ["P@10_TFIDF", "MRR_TFIDF", "P@10_Count", "MRR_Count"]
        ].mean()
    )


if __name__ == "__main__":
    main()
