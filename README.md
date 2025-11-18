# PDF QA Chatbot Pipeline (src-based)

## Features
- Upload a PDF and process it (clean, embed, index)
- Ask questions about the PDF via API (RAG)
- Health and stats endpoints
- Streamlit UI (optional)
- Dockerized for easy deployment

## Project Structure
```
src/
  api/           # FastAPI app and endpoints
  chatbot/       # Chatbot logic
  pipeline/      # Data cleaning, embedding, etc.
  data/          # PDF input files
  output/        # Cleaned text output
  vector_store/  # FAISS and embedding files
  config.py      # Centralized config
  requirements.txt
```

## Usage

### Local (dev)
1. Install dependencies:
   ```sh
   pip install -r src/requirements.txt
   ```
2. Run the pipeline and API:
   ```sh
   uvicorn src.api.main:app --reload
   ```
3. Upload a PDF via `/upload` endpoint or place it in `src/data/` and run the pipeline.

### Docker
1. Build the image:
   ```sh
   docker build -t pdf-qa-bot .
   ```
2. Run the container:
   ```sh
   docker run -p 8000:8000 pdf-qa-bot
   ```

## API Endpoints
- `POST /upload` — Upload a PDF and trigger the pipeline
- `POST /ask` — Ask a question about the PDF
- `GET /health` — Health check
- `GET /stats` — Document and system stats

## Environment
- Set your OpenAI API key in `.env` as `OPENAI_API_KEY=sk-...`


## Output
<img width="1208" height="893" alt="image" src="https://github.com/user-attachments/assets/fe718c4a-f737-4e18-9a07-7f266dbdc4dd" />

--- 
