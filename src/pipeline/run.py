import os
import logging
from pipeline.cleaner import extract_lines_from_pdf, clean_lines, save_cleaned_text
from pipeline.embed_and_index import embed_and_store

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(filename=None):
    # STEP 1: Load the PDF from the 'data/' folder
    try:
        data_folder = os.path.join(os.path.dirname(__file__), '../../data')
        data_folder = os.path.abspath(data_folder)
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]

        if not pdf_files:
            raise FileNotFoundError("❌ No PDF file found in the 'data/' folder!")

        if filename and filename in pdf_files:
            pdf_file = filename
        else:
            pdf_file = pdf_files[0]

        PDF_PATH = os.path.join(data_folder, pdf_file)
        output_folder = os.path.join(os.path.dirname(__file__), '../../output')
        output_folder = os.path.abspath(output_folder)
        OUTPUT_PATH = os.path.join(output_folder, f'cleaned_{pdf_file.replace(".pdf", ".txt")}')
        logger.info(f"📄 Using file: {PDF_PATH}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # STEP 2: Extract raw lines from the PDF
    try:
        lines = extract_lines_from_pdf(PDF_PATH)
        if not lines:
            raise ValueError("⚠️ No extractable text found in the PDF.")
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return

    # STEP 3: Clean the extracted lines
    cleaned = clean_lines(lines)

    # STEP 5: Save the cleaned output to disk
    try:
        save_cleaned_text(cleaned, OUTPUT_PATH)
        logger.info(f"✅ Cleaned file saved at: {OUTPUT_PATH}")
        # STEP 6: Run embedding and indexing
        embed_and_store()
    except Exception as e:
        logger.error(f"Error saving cleaned file: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # You can add custom CLI argument handling here if needed
        pass
    else:
        main()