import os
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from utils import ArabicLegalNormalizer

ARTIFACTS_DIR = "artifacts"

# if you dont have the embedding model on local sys replace and paste this : BAAI/bge-m3
class LegalSearchEngine:
    def __init__(self, model_name: str = r"D:\models\embeding_models\BAAI--bge-m3"):
        print("Loading Search Engine Resources...")

        # 1. Load embedding model (used only for query encoding)
        self.model = SentenceTransformer(model_name)
        self.query_instruction = "Represent this sentence for searching relevant documents: "

        # 2. Load main data (documents & metadata)
        self.df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "documents.parquet"))

        # 3. Load FAISS index
        self.faiss_index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "faiss.index"))

        # 4. Load BM25 index
        with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)

        self.normalizer = ArabicLegalNormalizer()
        print("Engine Ready 🚀")

    def encode_query(self, query: str):
        # Initial query normalization (important)
        # Note: Normalizing the query text is critical for BM25,
        # and can also be beneficial for dense models.
        # Here, the raw text is passed to the dense model since
        # language models are typically trained on raw text.
        # You may also experiment with normalized input.
        text = f"{self.query_instruction}{query}"
        embedding = self.model.encode([text], normalize_embeddings=True)
        return np.array(embedding, dtype='float32')

    def search_dense(self, query: str, k: int):
        q_emb = self.encode_query(query)
        scores, indices = self.faiss_index.search(q_emb, k)
        return scores[0], indices[0]

    def search_sparse(self, query: str, k: int):
        # For BM25, the query must be normalized and tokenized
        normalized_query = self.normalizer.normalize(query)
        tokenized_query = normalized_query.split()

        all_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(all_scores)[::-1][:k]
        top_k_scores = all_scores[top_k_indices]
        return top_k_scores, top_k_indices

    @staticmethod
    def min_max_normalize(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s == min_s:
            return np.ones_like(scores) if max_s > 0 else np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def apply_metadata_boosting(self, indices, scores, query):
        query_norm = self.normalizer.normalize(query)
        boosted_scores = scores.copy()

        for i, idx in enumerate(indices):
            meta = self.df.iloc[idx]['metadata']
            law_type = self.normalizer.normalize(meta.get('law_type', ''))

            # Boost: law type match
            if "مرسوم" in query_norm and "مرسوم" in law_type:
                boosted_scores[i] += 0.15
            elif "قرار" in query_norm and "قرار" in law_type:
                boosted_scores[i] += 0.15

            # Boost: more recent laws
            try:
                if int(meta.get('year', 0)) > 2020:
                    boosted_scores[i] += 0.05
            except:
                pass

        return boosted_scores

    def search(self, query: str, k: int = 10, alpha: float = 0.6) -> List[Dict]:
        retrieve_k = k * 2

        # 1. Parallel retrieval (serial in code, parallel conceptually)
        dense_scores, dense_indices = self.search_dense(query, k=retrieve_k)
        sparse_scores, sparse_indices = self.search_sparse(query, k=retrieve_k)

        # 2. Score normalization
        norm_dense = self.min_max_normalize(dense_scores)
        norm_sparse = self.min_max_normalize(sparse_scores)

        # 3. Result fusion (weighted sum / reciprocal rank fusion style)
        score_map = {}
        for idx, score in zip(dense_indices, norm_dense):
            score_map[idx] = score_map.get(idx, {'d': 0, 's': 0})
            score_map[idx]['d'] = score

        for idx, score in zip(sparse_indices, norm_sparse):
            score_map[idx] = score_map.get(idx, {'d': 0, 's': 0})
            score_map[idx]['s'] = score

        beta = 1.0 - alpha
        final_candidates = []
        for idx, scores in score_map.items():
            hybrid_score = (alpha * scores['d']) + (beta * scores['s'])
            final_candidates.append((idx, hybrid_score))

        # Initial sorting
        final_candidates.sort(key=lambda x: x[1], reverse=True)

        # Separate indices and scores for boosting
        cand_indices = np.array([x[0] for x in final_candidates])
        cand_scores = np.array([x[1] for x in final_candidates])

        # 4. Apply metadata-based boosting
        final_scores = self.apply_metadata_boosting(cand_indices, cand_scores, query)

        # 5. Final output
        results = []
        # Re-zip and select Top-K
        for idx, score in sorted(zip(cand_indices, final_scores), key=lambda x: x[1], reverse=True)[:k]:
            row = self.df.iloc[idx]
            meta = row['metadata']
            results.append({
                "chunk": row['chunk'],
                "title": meta.get('title'),
                "law_type": meta.get('law_type'),
                "id": meta.get('doc_id', 'N/A'),  # Added
                "link": meta.get('canonical_link', '#'),
                "score": round(float(score), 4)
            })

        return results

# questions = [
#     "ماذا يقرر القرار رقم ١ / ٢٠١٥ بشأن أحكام القرار رقم ١٣٣ / ٢٠٠٨؟",
#     "من يفوض لتوقيع اتفاقية الخدمات الجوية بين عمان وأوغندا وفق المرسوم رقم ٦٣ / ٩٥؟",
#     "ما هي شروط إنشاء الكليات والمعاهد العليا الخاصة وفق المرسوم رقم ٤٢ / ٩٩؟",
#     "من تم تعيينهم في السلك الدبلوماسي وفق المرسوم رقم ٢٠ / ٢٠١٨؟",
#     "ما التعديل الذي أُدخل على القرار رقم ٢ / ٢٠٠٩ بشأن تثمين الأراضي وفق القرار الوزاري رقم ٣٥ / ٢٠١٨؟",
#     "ما اختصاصات وزارة البلديات الإقليمية والبيئة وفق المرسوم رقم ١٨ / ٩٩؟",
#     "ما المواصفتان القياسيتان الدوليتان المعتمدتان كمواصفات عمانية ملزمة وفق القرار الوزاري رقم ١٣٨ / ٢٠١٩؟",
#     "ما الحكم بشأن استيراد الطيور الحية من هولندا وفق القرار الوزاري رقم ٣٨٣ / ٢٠١٧؟",
#     "ما التعديلات على اسم المديرية العامة لموارد المياه والري وفق المرسوم رقم ٩٣ / ٨٦؟",
#     "ما نظام الترخيص والعمل في المنطقة الاقتصادية الخاصة بالدقم وفق القرار رقم ٣٠ / ٢٠١٧؟"
# ]

if __name__ == "__main__":

    engine = LegalSearchEngine()

    user_query = "ماذا يقرر القرار رقم ١ / ٢٠١٥ بشأن أحكام القرار رقم ١٣٣ / ٢٠٠٨؟"
    results = engine.search(user_query)

    print(f"\n: {user_query}")
    for res in results:
        print("-" * 30)
        print(f"Score: {res['score']}")
        print(f"Title: {res['title']}")
        print(f"Text: {res['chunk']}")
