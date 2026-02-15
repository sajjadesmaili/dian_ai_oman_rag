import os
from openai import OpenAI
from search import LegalSearchEngine
from dotenv import load_dotenv



class OmanLawAssistant:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o"):
        self.search_engine = LegalSearchEngine()

        # Configure API Key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found!")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def format_references(self, results: list) -> str:
        """
        Build the references section exactly according to the user-defined format
        """
        if not results:
            return ""

        ref_text = "\n\n---\n**المصادر (References):**\n"
        for i, res in enumerate(results, 1):
            title = res.get('title', 'N/A')
            doc_id = res.get('id', 'N/A')
            link = res.get('link', '#')

            # Format: index - title | ID | link
            ref_text += f"{i}. **العنوان:** {title} | **ID:** {doc_id} | **الرابط:** {link}\n"

        return ref_text

    def generate_answer(self, query: str, top_k: int = 5):
        # 1. Perform search
        print(f"Searching for: {query}")
        results = self.search_engine.search(query, k=top_k)

        if not results:
            return "عذراً، لم يتم العثور على نتائج مطابقة في قاعدة البيانات."

        # 2. Prepare context for the LLM
        context_str = ""
        for i, doc in enumerate(results, 1):
            context_str += f"المستند {i}:\n{doc['chunk']}\n\n"

        # System prompt
        # Note: The model is instructed not to list references,
        # as they are appended separately in a controlled format.
        system_prompt = (
            "أنت مساعد قانوني ذكي متخصص في القوانين العمانية. "
            "أجب على سؤال المستخدم بناءً على السياق (Context) المقدم فقط. "
            "كن دقيقاً ومباشراً. لا تقم بإدراج قائمة المصادر أو الروابط في نهاية إجابتك، "
            "فقط أجب على السؤال واشرح المواد القانونية."
        )

        user_prompt = (
            f"السياق القانوني:\n{context_str}\n\n"
            f"السؤال: {query}\n\n"
            "الجواب:"
        )

        # 3. Send request to the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Low temperature for legal-grade responses
                max_tokens=1500
            )

            llm_text = response.choices[0].message.content.strip()

            # 4. Append formatted references to the final answer
            final_output = llm_text + self.format_references(results)

            return final_output

        except Exception as e:
            return f"Error: {str(e)}"


# --- Execution & Testing ---
if __name__ == "__main__":

    load_dotenv()
    api_key = os.getenv("API_KEY")
    bot = OmanLawAssistant(api_key=api_key)

    question = "ماذا يقرر القرار رقم ١ / ٢٠١٥ بشأن أحكام القرار رقم ١٣٣ / ٢٠٠٨؟"

    answer = bot.generate_answer(question)
    print(answer)
