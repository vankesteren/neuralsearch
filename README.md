# Semantic search

This repo implements neural / embedding based semantic search on a database of products.

# Usage
Installation:

`uv sync`

Index creation:

`uv run create_index.py`

Search CLI app:

`uv run search.py --k 10`

(k is the number of returned results)