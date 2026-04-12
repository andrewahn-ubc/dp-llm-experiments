#!/usr/bin/env python3
"""
Cluster prompts in a CSV based on embedding similarity.

What it does:
1. Loads a CSV file.
2. Reads the text from a specified column (default: "goal").
3. Computes sentence embeddings for each prompt.
4. Computes pairwise cosine similarities.
5. Clusters the prompts using agglomerative clustering.
6. Writes cluster IDs back into the same CSV in a new column.

Example:
    python cluster_prompts.py --csv train.csv

Optional:
    python cluster_prompts.py --csv train.csv --text-column goal --output-column goal_cluster --distance-threshold 0.35
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster prompts in a CSV using embedding similarity.")
    parser.add_argument(
        "--csv",
        type=str,
        default="train.csv",
        help="Path to the CSV file. Default: train.csv",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="goal",
        help='Name of the text column to cluster. Default: "goal"',
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="goal_cluster",
        help='Name of the output cluster column. Default: "goal_cluster"',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.88,
        help=(
            "Cosine distance threshold for agglomerative clustering. "
            "Smaller => more/smaller clusters, larger => fewer/bigger clusters. "
            "Default: 0.35"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding computation. Default: 64",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak backup of the original CSV before overwriting.",
    )
    return parser.parse_args()


def load_texts(df: pd.DataFrame, text_column: str) -> List[str]:
    if text_column not in df.columns:
        raise ValueError(f'Column "{text_column}" not found in CSV. Available columns: {list(df.columns)}')

    texts = df[text_column].fillna("").astype(str).tolist()

    if len(texts) == 0:
        raise ValueError("CSV is empty.")

    return texts


def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity
    )

    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    # With normalized embeddings, cosine similarity works nicely.
    # AgglomerativeClustering with metric="cosine" uses cosine distance = 1 - cosine similarity.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    labels = clustering.fit_predict(embeddings)
    return labels


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f'Error: file "{args.csv}" does not exist.', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    texts = load_texts(df, args.text_column)

    print(f"Loaded {len(texts)} rows from {args.csv}")
    print(f'Computing embeddings for column "{args.text_column}" using model "{args.model}"...')

    embeddings = compute_embeddings(
        texts=texts,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    print("Computing pairwise cosine similarities...")
    sim_matrix = cosine_similarity(embeddings)
    avg_offdiag_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / max(1, sim_matrix.size - len(sim_matrix))
    print(f"Average pairwise cosine similarity: {avg_offdiag_sim:.4f}")

    print(f"Clustering with distance threshold = {args.distance_threshold} ...")
    labels = cluster_embeddings(
        embeddings=embeddings,
        distance_threshold=args.distance_threshold,
    )

    df[args.output_column] = labels

    n_clusters = len(np.unique(labels))
    print(f"Found {n_clusters} clusters.")

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster_id, size in cluster_sizes.items():
        print(f"  Cluster {cluster_id}: {size} prompts")

    if args.backup:
        backup_path = args.csv + ".bak"
        shutil.copy2(args.csv, backup_path)
        print(f'Backup written to "{backup_path}"')

    df.to_csv(args.csv, index=False)
    print(f'\nUpdated CSV written back to "{args.csv}" with new column "{args.output_column}"')


if __name__ == "__main__":
    main()