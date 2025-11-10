# AI_UseCase with PDF Upload

Streamlit demo with OpenAI chat, simple RAG, optional web search, and **PDF ingestion**.

## Quickstart

1. Clone or unzip the project.
2. Create a virtual environment and install:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Configure environment:
   ```bash
   cp .env.example .env
   # edit .env to add OPENAI_API_KEY
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

## Environment variables

Copy `.env.example` to `.env` and set:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_SEARCH_MODEL=gpt-4o-mini-search-preview
RAG_TOP_K=4
RAG_CHUNK_SIZE=800
DEFAULT_SYSTEM_PROMPT=You are a helpful assistant. Use uploaded docs and cite filenames when useful.
```

## Features

- Upload **PDF** and **TXT** files from the sidebar. Text is extracted (PyPDF2) and chunked.
- In-memory embeddings store. No external DB.
- Retrieval-Augmented responses with optional web search.
- System prompt editor. One-click reset for chat and KB.

## Notes

- Web search calls an OpenAI *search-preview* model. You can disable it in the sidebar.
- Replace models in `.env` if your account lacks access to defaults.


## New features

- **Response Modes:** Toggle between *Concise* and *Detailed* in the sidebar.

- **Thinking indicator:** While the model is generating, the chat shows a transient “thinking…” placeholder and a spinner.

