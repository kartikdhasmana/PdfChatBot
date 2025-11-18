import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model once - moved to lazy loading
_model = None

def get_model():
    """Lazy load model to save memory"""
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _model

def chunk_text(text, chunk_size=600, chunk_overlap=150):
    """
    Breaks the cleaned document into smart chunks with context preservation.
    
    Uses hierarchical splitting: paragraphs → sentences → words
    to maintain semantic coherence.
    
    Parameters:
    - text: The full cleaned document as a string
    - chunk_size: Maximum number of characters in one chunk
    - chunk_overlap: Characters to overlap between chunks for context
    
    Returns:
    - List of text chunks with metadata
    """
    if not text.strip():
        raise ValueError("Input text is empty")
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],  # Better separators
        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks from text")

    # Add metadata to chunks
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_metadata.append({
            'chunk_id': i,
            'text': chunk,
            'char_count': len(chunk),
            'word_count': len(chunk.split())
        })

    return chunk_metadata

def create_hybrid_index(embeddings):
    """Create both L2 and cosine similarity indices with better performance"""
    dim = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    # Choose index type based on dataset size for better performance
    if n_vectors < 1000:
        # Flat indices for small datasets
        l2_index = faiss.IndexFlatL2(dim)
        cosine_index = faiss.IndexFlatIP(dim)
    else:
        # IVF indices for larger datasets
        nlist = min(int(np.sqrt(n_vectors)), 256)
        l2_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
        cosine_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist)
        
        # Train indices
        l2_index.train(embeddings.astype(np.float32))
        cosine_index.train(embeddings.astype(np.float32))
    
    # Add embeddings (convert to float32 to save memory)
    l2_index.add(embeddings.astype(np.float32))
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine_index.add(normalized_embeddings.astype(np.float32))
    
    return l2_index, cosine_index, normalized_embeddings

def embed_and_store(filename=None):
    """Main function to embed and store document chunks with improved error handling"""
    try:
        output_folder = os.path.join(os.path.dirname(__file__), '../../output')
        output_folder = os.path.abspath(output_folder)
        
        # Check if output folder exists
        if not os.path.exists(output_folder):
            raise FileNotFoundError(f"Output folder '{output_folder}' not found")
        
        # Find text files (exclude metadata)
        txt_files = [f for f in os.listdir(output_folder) if f.lower().endswith(".txt") and not f.endswith("_metadata.txt")]
        if not txt_files:
            raise FileNotFoundError("No cleaned .txt file found in 'output/' folder")
        
        # Use the file matching the PDF if provided
        if filename:
            cleaned_filename = f"cleaned_{filename.replace('.pdf', '.txt')}"
            if cleaned_filename in txt_files:
                cleaned_path = os.path.join(output_folder, cleaned_filename)
            else:
                raise FileNotFoundError(f"Cleaned file for {filename} not found in output folder.")
        else:
            cleaned_path = os.path.join(output_folder, txt_files[0])
        logger.info(f"Using cleaned file: {cleaned_path}")
        
        # Read the cleaned text
        with open(cleaned_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            raise ValueError("The cleaned text file is empty")
        
        logger.info(f"Loaded text with {len(text)} characters")
        
        # Create chunks with metadata
        chunk_metadata = chunk_text(text)
        chunks = [chunk['text'] for chunk in chunk_metadata]
        
        # Generate embeddings with optimized batching
        logger.info("Generating embeddings...")
        bge = get_model()  # Use lazy loading
        
        # Process in larger batches for better performance
        batch_size = 8 # Increased from 2
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = bge.encode(
                [f"Represent this sentence for retrieval: {chunk}" for chunk in batch],
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(batch),
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
            
            # Periodic garbage collection to manage memory
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        embeddings = np.vstack(all_embeddings)
        
        # Create vector store directory
        vector_store_dir = os.path.join(os.path.dirname(__file__), '../../vector_store')
        vector_store_dir = os.path.abspath(vector_store_dir)
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Create hybrid indices
        l2_index, cosine_index, normalized_embeddings = create_hybrid_index(embeddings)
        
        # Save indices
        faiss.write_index(l2_index, os.path.join(vector_store_dir, "l2_index.faiss"))
        faiss.write_index(cosine_index, os.path.join(vector_store_dir, "cosine_index.faiss"))
        
        # Save chunks and metadata
        with open(os.path.join(vector_store_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        
        with open(os.path.join(vector_store_dir, "chunk_metadata.pkl"), "wb") as f:
            pickle.dump(chunk_metadata, f)
        
        # Save embeddings (use float32 to save space)
        np.save(os.path.join(vector_store_dir, "embeddings.npy"), embeddings.astype(np.float32))
        np.save(os.path.join(vector_store_dir, "normalized_embeddings.npy"), normalized_embeddings.astype(np.float32))
        
        # Save configuration
        config = {
            'chunk_size': 600,
            'chunk_overlap': 150,
            'embedding_model': 'BAAI/bge-large-en-v1.5',
            'total_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1]
        }
        
        with open(os.path.join(vector_store_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        logger.info("✅ Embedding and indexing complete")
        logger.info(f"Created {len(chunks)} chunks with {embeddings.shape[1]} dimensional embeddings")
        
    except Exception as e:
        logger.error(f"Error during embedding and indexing: {e}")
        raise

if __name__ == "__main__":
    embed_and_store()