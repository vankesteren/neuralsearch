"""Script to run searches in your CLI."""

import sys
from pathlib import Path

import faiss
import polars as pl
import torch
from autocorrect import Speller
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from transformers.utils.logging import disable_progress_bar

## INITIALIZATION ##

if not Path("data/index.faiss").exists():
    print("Build index first with uv run create_index.py")
    sys.exit(1)

# init app
app = FastAPI()

# init transformer model and autocorrect
disable_progress_bar()
model = SentenceTransformer("data/embedder.torch")
speller = Speller(fast=True)

# init index and data
index = faiss.read_index("data/index.faiss")
df = pl.read_csv("data/products.csv")


def encode(input: str | list[str]) -> torch.Tensor:
    if isinstance(input, list):
        if len(input) == 1:
            return encode(input[0])
        return model.encode(input)
    return model.encode(input).reshape(1, -1)


async def encode_async(input: str | list[str]) -> torch.Tensor:
    """Create embedding of a (list of) string(s)."""
    if isinstance(input, list):
        if len(input) == 1:
            return encode_async(input[0])
        return await run_in_threadpool(model.encode, input)
    res = await run_in_threadpool(model.encode, input)
    return res.reshape(1, -1)


@app.get("/search")
async def search(query: str, k: int = 5) -> list[dict]:
    """Perform a search query on the products database."""
    q_corrected = speller(query)
    if q_corrected != query:
        print(f"💡 Also showing results for: {q_corrected}")
        e = await encode_async([query, q_corrected])
    else:
        e = await encode_async(query)
    dist, idx = index.search(e, k)
    res = (
        df[idx.flatten()]
        .with_columns(distance=dist.flatten())
        .sort("distance")
        .unique(subset="product_id", keep="first")
        .slice(0, k)
    )
    return JSONResponse(res.to_dicts())
