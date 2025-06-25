# matcher.py
from sentence_transformers import SentenceTransformer
import numpy as np
from thefuzz import process

# Load the model only once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts: list[str]) -> np.ndarray:
    """Embeds a list of texts into vectors."""
    return model.encode(texts, convert_to_numpy=True)

def search_top_k(index, query_vector: np.ndarray, k: int = 5):
    """Searches a FAISS index for the top k most similar vectors."""
    distances, indices = index.search(query_vector, k)
    return indices[0], distances[0]

def fuzzy_match(query: str, choices: list[str], limit: int = 5):
    """Performs fuzzy string matching."""
    return process.extract(query, choices, limit=limit)