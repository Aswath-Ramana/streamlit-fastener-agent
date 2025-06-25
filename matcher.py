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

def fuzzy_match(query: str, choices: dict, limit: int = 5, score_cutoff: int = 60):
    """
    Performs fuzzy string matching with a score cutoff.
    'choices' should be a dictionary for this to work correctly and return indices.
    """
    # The process.extract function with a dictionary automatically handles the score_cutoff
    return process.extract(query, choices, limit=limit, score_cutoff=score_cutoff)