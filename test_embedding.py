from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

embedding = model.encode("hello world")

print(len(embedding))
print(embedding[:5])