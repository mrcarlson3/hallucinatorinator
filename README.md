# Hallucinatorinator

Detect fabricated legal claims using local LLMs and optional CourtListener API verification. Attempts to evaluate the validity of legal statements and check citations against a legal database.

## Quick Start

```bash
# Install Ollama
brew install ollama

# Download model
ollama pull llama3:8b

# Run detector 
python3 detector.py


```

## Features

- **detector.py** - Standalone hallucination detection using local LLM
- **legal_rag.py** - Citation verification via CourtListener API 
- 100% local operation (apart from  CourtListener API calls)
- Rate limiting and input sanitization
- Audit logging


## Disclaimer

Educational purposes only.

