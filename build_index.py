import os
import pickle
import numpy as np
import pandas as pd
import faiss
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils import ArabicLegalNormalizer

# Path configuration
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# if you dont have the embedding model on local sys replace and paste this : BAAI/bge-m3
class IndexBuilder:
    def __init__(self, model_name: str = r"D:\models\embeding_models\BAAI--bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.doc_instruction = "Represent this document for retrieval: "
        self.normalizer = ArabicLegalNormalizer()

    def build_dense_index(self, texts: list):
        print("Generating Dense Embeddings...")
        # Add model-specific instruction (BGE instruction tuning)
        inputs = [f"{self.doc_instruction}{t}" for t in texts]

        embeddings = self.model.encode(inputs, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype='float32')

        # Build FAISS index (Inner Product for cosine similarity since embeddings are normalized)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def build_sparse_index(self, df: pd.DataFrame, title_weight: int = 3):
        print("Building BM25 Index...")
        corpus_tokens = []

        for _, row in df.iterrows():
            meta = row['metadata']
            chunk_text = row['chunk']

            # Normalization and tokenization
            title_tokens = self.normalizer.normalize(meta.get('title', '')).split()
            body_tokens = self.normalizer.normalize(chunk_text).split()

            # Apply higher weight to title tokens
            combined_tokens = (title_tokens * title_weight) + body_tokens
            corpus_tokens.append(combined_tokens)

        bm25 = BM25Okapi(corpus_tokens)
        return bm25

    def run(self, df: pd.DataFrame):
        # 1. Prepare text for dense embeddings
        print("Formatting documents...")
        formatted_docs = df.apply(ArabicLegalNormalizer.preprocess_for_embedding, axis=1).tolist()

        # 2. Build FAISS index
        faiss_index = self.build_dense_index(formatted_docs)

        # 3. Build BM25 index
        bm25_index = self.build_sparse_index(df)

        # 4. Persist artifacts
        print(f"Saving artifacts to {ARTIFACTS_DIR}...")

        # a) Save DataFrame (metadata)
        df.to_parquet(os.path.join(ARTIFACTS_DIR, "documents.parquet"))

        # b) Save FAISS index
        faiss.write_index(faiss_index, os.path.join(ARTIFACTS_DIR, "faiss.index"))

        # c) Save BM25 index using pickle
        with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25_index, f)

        print("Indexing Complete ✅")


if __name__ == "__main__":

    file_path = r"D:\progam\Laqaee\oman_laws_final_dataset.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    builder = IndexBuilder()
    builder.run(df)
