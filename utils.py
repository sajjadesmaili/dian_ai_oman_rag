import re
import pandas as pd


class ArabicLegalNormalizer:
    """
    Handles text normalization specifically for Arabic search.
    Shared between Indexing and Searching phases.
    """

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Remove Tatweel (Kashida)
        text = re.sub(r'ـ', '', text)

        # Remove Diacritics (Tashkeel)
        text = re.sub(r'[\u064B-\u065F]', '', text)

        # Unify Alef forms (أ, إ, آ -> ا)
        text = re.sub(r'[أإآ]', 'ا', text)

        # Normalize Ta Marbuta (ة -> ه)
        text = re.sub(r'ة', 'ه', text)

        # Normalize Ya (ي -> ى)
        text = re.sub(r'ي$', 'ى', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def preprocess_for_embedding(row: pd.Series) -> str:
        """
        Constructs a semantically rich representation.
        """
        meta = row['metadata']
        chunk_text = row['chunk']

        clean_text = re.sub(r'\[.*?\]', '', chunk_text).strip()

        formatted_text = (
            f"القانون: {meta.get('law_type', 'غير محدد')}\n"
            f"الرقم: {meta.get('law_number', '')}\n"
            f"السنة: {meta.get('year', '')}\n"
            f"العنوان: {meta.get('title', '')}\n"
            f"النص: {clean_text}"
        )
        return formatted_text