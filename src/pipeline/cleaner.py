import os
import pdfplumber
from collections import Counter
import re
import logging
from typing import List, Dict
from pipeline.eda_tools import remove_unwanted_patterns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_lines_from_pdf(pdf_path: str) -> List[str]:
    """Extract text and table lines from PDF with better error handling"""
    lines = []
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return lines
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages from {pdf_path}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_lines = []
                
                # Extract text lines
                text = page.extract_text()
                if text:
                    # Split and filter empty lines in one step
                    text_lines = [line.strip() for line in text.split('\n') if line.strip()]
                    page_lines.extend(text_lines)
                
                # Extract tables more efficiently
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if table:  # Check if table is not empty
                            page_lines.append('---TABLE---')
                            # Process rows more efficiently
                            for row in table:
                                if row:  # Check if row is not empty
                                    # Use list comprehension for better performance
                                    cleaned_row = [cell.strip() if cell else '' for cell in row]
                                    if any(cleaned_row):  # Only add non-empty rows
                                        page_lines.append('\t'.join(cleaned_row))
                
                if page_lines:
                    lines.extend(page_lines)
                elif page_num % 10 == 0:  # Log warning every 10 empty pages
                    logger.warning(f"Page {page_num} is empty or image-only")
                    
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        raise
    
    logger.info(f"Extracted {len(lines)} lines from PDF")
    return lines

def enhanced_line_cleaning(lines: List[str]) -> List[str]:
    """No cleaning, just return lines as-is"""
    return lines

def clean_lines(lines: List[str]) -> List[str]:
    """Skip cleaning and return lines as-is"""
    logger.info(f"Skipping cleaning. Returning {len(lines)} lines as-is.")
    return lines

def save_cleaned_text(lines: List[str], output_path: str) -> None:
    """Save cleaned text and metadata with optimized I/O"""
    if not lines:
        logger.warning("No lines to save")
        return
        
    try:
        # Create output directory
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metadata once
        total_chars = sum(len(line) for line in lines)
        total_words = sum(len(line.split()) for line in lines)
        
        # Save main text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # Save metadata
        metadata = {
            'total_lines': len(lines),
            'total_chars': total_chars,
            'total_words': total_words,
            'avg_line_length': total_chars / len(lines) if lines else 0
        }
        
        meta_path = output_path.replace('.txt', '_metadata.txt')
        with open(meta_path, 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                formatted_key = key.replace('_', ' ').title()
                # Format numbers for better readability
                if isinstance(value, float):
                    f.write(f"{formatted_key}: {value:.2f}\n")
                else:
                    f.write(f"{formatted_key}: {value:,}\n")
        
        logger.info(f"Cleaned text saved to: {output_path}")
        logger.info(f"Metadata saved to: {meta_path}")
        logger.info(f"Processed {len(lines):,} lines, {total_words:,} words, {total_chars:,} characters")
        
    except Exception as e:
        logger.error(f"Could not save cleaned text: {e}")
        raise