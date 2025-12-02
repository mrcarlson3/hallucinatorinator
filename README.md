# Hallucinatorinator

Detect fabricated legal claims using local LLMs and optional CourtListener API verification.

## Quick Start

```bash
# Install Ollama
brew install ollama

# Download model
ollama pull llama3:8b

# Run detector (interactive mode)
python3 detector.py

# Run with RAG verification (requires API token)
export COURTLISTENER_TOKEN="your_token"
python3 legal_rag.py
```

## Features

- **detector.py** - Standalone hallucination detection using local LLM
- **legal_rag.py** - Citation verification via CourtListener API + vector DB
- 100% local operation (RAG optional)
- Rate limiting and input sanitization
- Audit logging

## Configuration

```python
# detector.py
MODEL = "llama3:8b"
MAX_REQUESTS_PER_MINUTE = 10

# legal_rag.py  
export COURTLISTENER_TOKEN="token"  # Optional
```

## Security

- Input/output sanitization
- Timeout protection (120s)
- All sensitive files in .gitignore

## Disclaimer

Educational purposes only. Not a substitute for professional legal review.

