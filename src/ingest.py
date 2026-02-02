from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/documents.txt", "r", encoding="utf-8") as file_handle:
    text_lines = [line.strip() for line in file_handle.readlines() if line.strip()]

vector_representations = embedder.encode(text_lines)

np.save("data/embeddings.npy", vector_representations)
np.save("data/texts.npy", np.array(text_lines, dtype=object))

print("Documents ingested successfully.")
print(f"Total documents stored: {len(text_lines)}")
