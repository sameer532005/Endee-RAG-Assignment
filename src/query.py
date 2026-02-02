from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

embedder = SentenceTransformer("all-MiniLM-L6-v2")

stored_vectors = np.load("data/embeddings.npy")
stored_texts = np.load("data/texts.npy", allow_pickle=True)

user_input = input("Enter your question: ")

query_vector = embedder.encode([user_input])[0]

similarity_scores = stored_vectors @ query_vector / (
    norm(stored_vectors, axis=1) * norm(query_vector)
)

top_match_index = similarity_scores.argmax()

print("\nRetrieved Context:")
print(stored_texts[top_match_index])

print("\nFinal Answer:")
print(stored_texts[top_match_index])
