"""
PDF QA Chatbot API and Pipeline Entrypoint

- /upload: Upload a PDF and trigger the pipeline (clean, embed, index)
- /ask: Ask questions about the processed PDF
- /health: Health check
- /stats: Document and system statistics

This file orchestrates the full pipeline and serves the FastAPI app.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os

# Add the 'src' directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from chatbot.logic import load_index, search_and_ask_gpt, search_and_ask_gpt_old_format
import logging
from typing import Optional, Dict, Any
import time
import shutil
from pipeline.run import main as pipeline_main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Question Answering API",
    description="API for asking questions about PDF documents using RAG",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove index loading at startup
# indices_data = load_index()

class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="The question to ask about the document")
    top_k: Optional[int] = Field(default=10, ge=1, le=20, description="Number of chunks to retrieve")

class QuestionResponse(BaseModel):
    answer: str
    processing_time: float
    chunks_used: int
    confidence: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Question Answering API",
        "version": "2.0.0",
        "endpoints": {
            "ask": "/ask - POST - Ask a question about the document",
            "health": "/health - GET - Check API health",
            "stats": "/stats - GET - Get document statistics"
        }
    }

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and trigger the pipeline."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(data_dir, exist_ok=True)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided for uploaded file.")
    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Run the pipeline in the background, passing the filename
    background_tasks.add_task(pipeline_main, file.filename)
    return {"message": f"File '{file.filename}' uploaded and pipeline started."}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: Request, payload: Question):
    """Ask a question about the processed PDF document."""
    import time as t
    start = t.time()
    try:
        indices_data = load_index()
        l2_index, cosine_index, chunks, chunk_metadata, config = indices_data
        answer = search_and_ask_gpt(
            question=payload.question,
            l2_index=l2_index,
            cosine_index=cosine_index,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            config=config,
            top_k=payload.top_k or 10
        )
        processing_time = t.time() - start
        return QuestionResponse(
            answer=answer,
            processing_time=processing_time,
            chunks_used=payload.top_k or 10,
            confidence=None
        )
    except FileNotFoundError as e:
        logger.error(f"Index not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        raise HTTPException(status_code=500, detail="Could not answer question. The index may not be ready. Please upload and process a PDF first.")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        indices_data = load_index()
        l2_index, cosine_index, chunks, chunk_metadata, config = indices_data
        chunk_count = len(chunks) if chunks else 0
        return {
            "status": "healthy",
            "chunks_loaded": chunk_count,
            "indices_loaded": l2_index is not None,
            "timestamp": time.time()
        }
    except FileNotFoundError as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e), "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.get("/stats")
async def get_stats():
    """Get document and system statistics"""
    try:
        indices_data = load_index()
        l2_index, cosine_index, chunks, chunk_metadata, config = indices_data
        stats = {
            "total_chunks": len(chunks) if chunks else 0,
            "index_type": "hybrid" if cosine_index else "l2_only",
            "has_metadata": chunk_metadata is not None,
            "configuration": config if config else "default"
        }
        if chunk_metadata:
            avg_chars = sum(c['char_count'] for c in chunk_metadata) / len(chunk_metadata)
            avg_words = sum(c['word_count'] for c in chunk_metadata) / len(chunk_metadata)
            stats["avg_chunk_length"] = {
                "chars": round(avg_chars, 2),
                "words": round(avg_words, 2)
            }
        else:
            avg_len = sum(len(c.split()) for c in chunks) / len(chunks)
            stats["avg_chunk_length"] = {
                "words": round(avg_len, 2)
            }
        return stats
    except FileNotFoundError as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to calculate stats: {e}")
        raise HTTPException(status_code=500, detail="Could not calculate stats. The index may not be ready. Please upload and process a PDF first.")