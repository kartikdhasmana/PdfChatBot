import os
import pickle
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import logging
from typing import List, Tuple, Optional
from langchain.memory import ConversationBufferMemory
from rapidfuzz import fuzz
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize models with lazy loading
_bge_model = None
_reranker_model = None
_openai_client = None

def get_bge_model():
    """Lazy load BGE model"""
    global _bge_model
    if _bge_model is None:
        _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _bge_model

def get_reranker_model():
    """Lazy load reranker model"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        logger.info("✅ Reranker model 'BAAI/bge-reranker-v2-m3' successfully loaded.")
    return _reranker_model

def get_openai_client():
    """Lazy load OpenAI client"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

# Memory for conversation context
memory = ConversationBufferMemory(return_messages=True)

def load_index():
    """Load vector indices and chunks with better error handling"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../vector_store'))
    
    files = {
        'l2_index': os.path.join(base_dir, "l2_index.faiss"),
        'cosine_index': os.path.join(base_dir, "cosine_index.faiss"),
        'chunks': os.path.join(base_dir, "chunks.pkl"),
        'metadata': os.path.join(base_dir, "chunk_metadata.pkl"),
        'config': os.path.join(base_dir, "config.pkl")
    }
    
    # Check all required files at once
    missing = [name for name, path in files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required index files: {', '.join(missing)}. Please upload and process a PDF first.")

    try:
        # Load indices
        l2_index = faiss.read_index(files['l2_index'])
        cosine_index = faiss.read_index(files['cosine_index'])
        
        # Load pickled data
        with open(files['chunks'], "rb") as f:
            chunks = pickle.load(f)
        with open(files['metadata'], "rb") as f:
            chunk_metadata = pickle.load(f)
        with open(files['config'], "rb") as f:
            config = pickle.load(f)
            
        logger.info(f"Loaded indices with {len(chunks)} chunks")
        return l2_index, cosine_index, chunks, chunk_metadata, config
        
    except Exception as e:
        logger.error(f"Error loading indices: {e}")
        raise

def is_financial_term(query: str) -> bool:
    """Check if query contains financial/business terms"""
    financial_terms = {
        'pat', 'profit after tax', 'revenue', 'sales', 'income', 'earnings', 'ebitda', 
        'cash flow', 'assets', 'liabilities', 'equity', 'debt', 'roi', 'roe', 'margin',
        'turnover', 'expenses', 'cost', 'investment', 'dividend', 'share', 'valuation',
        'growth', 'pbt', 'profit before tax', 'net profit', 'gross profit', 'operating profit',
        'balance sheet', 'p&l', 'financial', 'accounting', 'audit', 'tax', 'depreciation'
    }
    query_lower = query.lower()
    return any(term in query_lower for term in financial_terms)

def has_numbers_or_financials(chunk: str) -> bool:
    """Check if chunk contains numbers or financial indicators"""
    # Look for numbers, currency symbols, financial terms
    financial_indicators = [
        r'\d+\.?\d*\s*(?:crore|lakh|million|billion|thousand)',
        r'₹\s*\d+', r'\$\s*\d+', r'(?:rs\.?|rupees)\s*\d+',
        r'\d+\.?\d*\s*%', r'\d+\.?\d*\s*percent',
        r'(?:increase|decrease|growth|decline)\s*(?:of|by)?\s*\d+',
        r'(?:revenue|sales|profit|loss|income|expenses?|cost)\s*(?:of|is|was|were)?\s*[₹$]?\s*\d+'
    ]
    
    chunk_lower = chunk.lower()
    return any(re.search(pattern, chunk_lower) for pattern in financial_indicators)

def keyword_search(query: str, chunks: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
    """Enhanced keyword search with financial term prioritization"""
    keywords = query.lower().split()
    results = []
    
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        
        # Basic keyword score
        score = sum(1 for k in keywords if k in chunk_lower)
        
        # Boost score if chunk contains numbers/financials and query is financial
        if score > 0 and is_financial_term(query) and has_numbers_or_financials(chunk):
            score *= 2.0  # Double the score for financial data
            
        if score > 0:
            results.append((idx, float(score), chunk))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def fuzzy_keyword_search(query: str, chunks: List[str], threshold: int = 80, top_k: int = 10) -> List[Tuple[int, float, str]]:
    """Enhanced fuzzy keyword search with financial prioritization"""
    keywords = query.lower().split()
    results = []
    
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = max(fuzz.partial_ratio(k, chunk_lower) for k in keywords)
        
        # Boost score for financial chunks
        if score >= threshold and is_financial_term(query) and has_numbers_or_financials(chunk):
            score *= 1.2
            
        if score >= threshold:
            results.append((idx, float(score), chunk))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def advanced_similarity_search(query: str, l2_index, cosine_index, chunks, 
                             l2_top_k=15, cosine_top_k=15, final_top_k=15) -> List[Tuple[int, float, str]]:
    """Enhanced similarity search with financial data prioritization"""
    bge = get_bge_model()
    query_embedding = bge.encode([query], convert_to_numpy=True)
    
    # L2 search
    l2_dists, l2_inds = l2_index.search(query_embedding, l2_top_k)
    
    # Cosine search with normalized embedding
    normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    cos_scores, cos_inds = cosine_index.search(normalized, cosine_top_k)
    
    combined_scores = {}
    chunk_count = len(chunks)
    is_financial_query = is_financial_term(query)
    
    # Process L2 results
    for idx, dist in zip(l2_inds[0], l2_dists[0]):
        if 0 <= idx < chunk_count:
            score = 1 / (1 + dist)
            # Boost financial chunks for financial queries
            if is_financial_query and has_numbers_or_financials(chunks[idx]):
                score *= 1.5
            combined_scores[idx] = max(combined_scores.get(idx, 0), score)
    
    # Process cosine results
    for idx, score in zip(cos_inds[0], cos_scores[0]):
        if 0 <= idx < chunk_count:
            # Boost financial chunks for financial queries
            if is_financial_query and has_numbers_or_financials(chunks[idx]):
                score *= 1.5
            combined_scores[idx] = max(combined_scores.get(idx, 0), score)
    
    # Add enhanced keyword search results
    keyword_results = keyword_search(query, chunks, top_k=final_top_k)
    for idx, score, _ in keyword_results:
        combined_scores[idx] = max(combined_scores.get(idx, 0), score + 1.0)

    # Sort and return top results
    results = [(idx, score, chunks[idx]) for idx, score in combined_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:final_top_k]

def multi_pass_retrieval(query: str, l2_index, cosine_index, chunks, final_top_k=15) -> List[Tuple[int, float, str]]:
    """Enhanced multi-pass retrieval with document-specific focus"""
    # 1. Semantic search (L2 + cosine)
    bge = get_bge_model()
    query_embedding = bge.encode([query], convert_to_numpy=True)
    l2_dists, l2_inds = l2_index.search(query_embedding, 15)
    normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    cos_scores, cos_inds = cosine_index.search(normalized, 15)
    
    combined_scores = {}
    chunk_count = len(chunks)
    is_financial_query = is_financial_term(query)
    
    for idx, dist in zip(l2_inds[0], l2_dists[0]):
        if 0 <= idx < chunk_count:
            score = 1 / (1 + dist)
            if is_financial_query and has_numbers_or_financials(chunks[idx]):
                score *= 1.5
            combined_scores[idx] = max(combined_scores.get(idx, 0), score)
            
    for idx, score in zip(cos_inds[0], cos_scores[0]):
        if 0 <= idx < chunk_count:
            if is_financial_query and has_numbers_or_financials(chunks[idx]):
                score *= 1.5
            combined_scores[idx] = max(combined_scores.get(idx, 0), score)
    
    # 2. Enhanced keyword search
    keyword_results = keyword_search(query, chunks, top_k=final_top_k)
    for idx, score, _ in keyword_results:
        combined_scores[idx] = max(combined_scores.get(idx, 0), score + 1.0)
    
    # 3. Enhanced fuzzy keyword search
    fuzzy_results = fuzzy_keyword_search(query, chunks, threshold=80, top_k=final_top_k)
    for idx, score, _ in fuzzy_results:
        combined_scores[idx] = max(combined_scores.get(idx, 0), score / 100.0 + 0.5)
    
    # 4. Aggregate and sort
    results = [(idx, score, chunks[idx]) for idx, score in combined_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:final_top_k]

def hybrid_rerank(query: str, candidates: List[Tuple[int, float, str]], top_k=5) -> List[Tuple[int, float, str]]:
    """Enhanced reranking with financial data prioritization"""
    if not candidates:
        return []
    
    try:
        reranker = get_reranker_model()
        rerank_inputs = [[query, chunk] for _, _, chunk in candidates]
        rerank_scores = reranker.predict(rerank_inputs)
        
        # Apply financial boost to reranked scores
        is_financial_query = is_financial_term(query)
        for i, (_, _, chunk) in enumerate(candidates):
            if is_financial_query and has_numbers_or_financials(chunk):
                rerank_scores[i] *= 1.2  # Boost financial chunks
        
        # Combine with original candidates
        reranked = [(candidates[i][0], float(rerank_scores[i]), candidates[i][2]) 
                   for i in range(len(candidates))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
        
    except Exception as e:
        logger.error(f"BGE reranker failed: {e}, falling back to embedding similarity.")
        
        # Enhanced fallback with financial prioritization
        bge = get_bge_model()
        chunks_text = [chunk for _, _, chunk in candidates]
        
        query_emb = bge.encode([query], convert_to_numpy=True)
        candidate_embs = bge.encode(chunks_text, convert_to_numpy=True)
        sim_scores = np.dot(candidate_embs, query_emb.T).flatten()
        
        # Apply financial boost
        is_financial_query = is_financial_term(query)
        for i, (_, _, chunk) in enumerate(candidates):
            if is_financial_query and has_numbers_or_financials(chunk):
                sim_scores[i] *= 1.2
        
        reranked = [(candidates[i][0], float(sim_scores[i]), candidates[i][2]) 
                   for i in range(len(candidates))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

def detect_user_intent(query: str) -> str:
    """Enhanced intent detection that prioritizes document-specific requests"""
    # Always treat queries as document-specific first
    return "document_specific"

def expand_query_for_document(query: str) -> str:
    """Enhanced query expansion that emphasizes document-specific information"""
    if len(query.split()) > 7:
        return query
        
    try:
        # Improved prompt that emphasizes document-specific information
        prompt = f"""Rewrite this question to specifically ask for information from the uploaded document/company report. 
Focus on getting specific numbers, figures, and company-specific details rather than general definitions.

Examples:
- "what is PAT" → "what is the PAT (Profit After Tax) figure for this company and how has it changed"
- "what is revenue" → "what is the company's revenue and revenue breakdown from this document"
- "what is EBITDA" → "what is this company's EBITDA and how is it calculated in their financial statements"

Original question: {query}

Document-specific question:"""
        
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
            top_p=0.9
        )
        expanded = response.choices[0].message.content.strip()
        
        if expanded and expanded != query:
            logger.info(f"Expanded query: '{expanded}' (original: '{query}')")
            return expanded
            
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
    
    # Fallback: add document-specific context
    return f"From this document, {query} - provide specific numbers and company details"

def generate_document_specific_prompt(question: str, context_chunks: List[str]) -> str:
    """Enhanced prompt that forces document-specific answers with numbers"""
    context = "\n---\n".join(context_chunks)
    
    # Get recent conversation history
    recent_messages = memory.chat_memory.messages[-10:] if len(memory.chat_memory.messages) > 10 else memory.chat_memory.messages
    history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in recent_messages])

    return f"""You are an AI assistant analyzing a specific company document. Your job is to provide DOCUMENT-SPECIFIC answers with actual numbers, figures, and company details.

CRITICAL INSTRUCTIONS:
- **ALWAYS prioritize specific company data, numbers, figures, and metrics from the document**
- **Use all of the document chunks together** to generate a unified, detailed answer.
- Do not skip any relevant information across chunks unless it is clearly redundant.
- If the answer is not present in the document, say: "This information is not available in the provided document."
- If the answer is not present in the document, ONLY reply with: "This information is not available in the provided document." Do not attempt to answer otherwise.
- **NEVER give generic definitions** - users want to know about THIS specific company
- **Include actual values, percentages, amounts, and financial figures wherever available**
- Quote exact numbers, company names, time periods, and specific details
- If asked about financial terms (PAT, revenue, EBITDA, etc.), provide:
  1. The actual figures for this company
  2. Time periods (which year/quarter)
  3. Any trends or changes mentioned
  4. Comparisons with previous periods if available
- Synthesize information across all chunks to give a comprehensive company-specific answer
- Use a natural, informative writing style but focus on facts and figures

DOCUMENT CHUNKS:
\"\"\"{context}\"\"\"

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

COMPANY-SPECIFIC ANSWER WITH NUMBERS AND DETAILS:"""

def search_and_ask_gpt(
    question: str,
    l2_index: faiss.Index,
    cosine_index: faiss.Index,
    chunks: List[str],
    chunk_metadata: Optional[List] = None,
    config: Optional[dict] = None,
    top_k: int = 10  # Increased to get more financial data
) -> str:
    """Enhanced QA function focused on document-specific responses"""
    try:
        # Expand query for document-specific information
        expanded_question = expand_query_for_document(question)
        
        # Search for relevant chunks with financial prioritization
        results = multi_pass_retrieval(expanded_question, l2_index, cosine_index, chunks, final_top_k=top_k*3)
        if not results:
            return "No relevant information found for your query."
        
        # Enhanced reranking with financial prioritization
        selected_chunks = hybrid_rerank(expanded_question, results, top_k=top_k)
        if not selected_chunks:
            return "No relevant information found for your query."

        # Extract chunk texts
        top_chunks = [chunk for _, _, chunk in selected_chunks]

        logger.info("📄 Final chunks used for prompt:")
        for i, chunk in enumerate(top_chunks, 1):
            logger.info(f"{i}. {chunk[:100]}...")

        # Add to conversation memory
        memory.chat_memory.add_user_message(question)

        # Generate document-specific prompt
        prompt = generate_document_specific_prompt(question, top_chunks)

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=1500,  # Increased for more detailed responses
            top_p=0.9
        )

        answer = response.choices[0].message.content.strip()
        memory.chat_memory.add_ai_message(answer)
        return answer

    except Exception as e:
        logger.error(f"Error in question answering: {e}")
        return f"Sorry, an error occurred: {e}"

def search_and_ask_gpt_old_format(question: str, index: faiss.Index, chunks: List[str], top_k: int = 5) -> str:
    """Legacy function for backward compatibility"""
    return search_and_ask_gpt(question, index, None, chunks, None, None, top_k)