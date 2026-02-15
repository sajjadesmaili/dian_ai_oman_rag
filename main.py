import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import OmanLawAssistant


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Global variable to hold the RAG engine instance
rag_instance = None


# --- Application Lifespan Management ---
# This ensures models and indexes are loaded only once at server startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance
    print("⏳ Starting up: Loading RAG models and indexes...")
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API Key is missing!")

        # Initialize the RAG engine
        rag_instance = OmanLawAssistant(api_key=OPENAI_API_KEY)
        print("✅ System Ready! API is running.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

    yield

    print("🛑 Shutting down...")
    # Close database connections or release resources here if needed


# --- FastAPI Application ---
app = FastAPI(
    title="Oman Law RAG API",
    description="API for querying Oman Laws",
    version="1.0.0",
    lifespan=lifespan
)


# --- Request and Response Models (Pydantic) ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Default: retrieve top 5 documents


class QueryResponse(BaseModel):
    query: str
    answer: str
    processing_time: float


# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to Oman Law AI API. Go to /docs to test."}


@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    This endpoint takes the user’s question and returns the LLM’s answer along with the sources.
    """
    if not rag_instance:
        raise HTTPException(status_code=503, detail="System is still loading or failed to initialize.")

    start_time = time.time()

    try:
        # Call the main method from the LLM/RAG layer
        # This method returns the final formatted answer along with sources
        result_text = rag_instance.generate_answer(
            query=request.query,
            top_k=request.top_k
        )

        process_time = round(time.time() - start_time, 2)

        return QueryResponse(
            query=request.query,
            answer=result_text,
            processing_time=process_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
