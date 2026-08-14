import math

import faiss
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# read data
df = pl.read_csv("data/products.csv")

# init transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model.save("data/embedder.torch")

# init index
index = faiss.IndexFlatL2(384)

# encode batches and add to index
batch_size = 250
for batch in tqdm(df.iter_slices(n_rows=batch_size), total=math.ceil(len(df) / batch_size)):
    items_list = batch.get_column("product_name").to_list()
    embeddings = model.encode(items_list, convert_to_tensor=True)
    index.add(embeddings)

faiss.write_index(index, "data/index.faiss")
