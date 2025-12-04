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
In order to get courtlistener verification, set your CourtListener API key as an env variable:

```bash
export COURTLISTENER_API_KEY="your_api_key_here"
```
Must make an account on https://www.courtlistener.com/ to get an API key.
## Features

- **detector.py** - Standalone hallucination detection using local LLM
- **legal_rag.py** - Citation verification via CourtListener API 
- 100% local operation (apart from  CourtListener API calls)
- Rate limiting and input sanitization
- Audit logging


## Disclaimer

Educational purposes only.

