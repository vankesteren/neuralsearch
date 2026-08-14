"""Script to run searches in your CLI."""

import sys
from pathlib import Path

import faiss
import polars as pl
import typer
from autocorrect import Speller
from sentence_transformers import SentenceTransformer
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

if not Path("data/index.faiss").exists():
    print("Build index first with uv run create_index.py")
    sys.exit(1)

# init transformer model
model = SentenceTransformer("data/embedder.torch")
index = faiss.read_index("data/index.faiss")
df = pl.read_csv("data/products.csv")
speller = Speller(fast=True)


def search(query: str, k: int = 5) -> pl.DataFrame:
    q_corrected = speller(query)
    if q_corrected != query:
        print(f"💡 Also showing results for: {q_corrected}")
        e = model.encode([query, q_corrected])
    else:
        e = model.encode(query).reshape(1, -1)
    dist, idx = index.search(e, k)
    res = df[idx.flatten()].with_columns(distance=dist.flatten())
    return res.sort("distance").unique(subset="product_id", keep="first").slice(0, 10)


def main(k: int = 5):
    while True:
        q = typer.prompt("🔎 Search query")

        print(search(q, k))


if __name__ == "__main__":
    typer.run(main)
