# Hallucinatorinator 🔍⚖️

Detect fabricated legal claims, fake cases, and invented statutes using a local LLM. Zero external API calls, fully private operation.

## Features

- Identifies hallucinated legal citations and statutes
- Verifies claims using logical reasoning
- 100% local operation (no cloud APIs)
- Built-in rate limiting and input sanitization
- Comprehensive audit logging


The script analyzes the sample text and outputs a JSON report of detected hallucinations.

## Configuration

Edit `detector.py` to customize:

```python
MODEL = "llama3:8b"      # Change model
MAX_INPUT_LENGTH = 50000           # Max text size (50KB)
TIMEOUT_SECONDS = 120              # Inference timeout
MAX_REQUESTS_PER_MINUTE = 10       # Rate limit
```

## Security

- **No external calls** - All processing is local
- **Rate limiting** - Prevents abuse (10 req/min default)
- **Input sanitization** - 50KB max, type validation
- **Output sanitization** - Control character removal
- **Timeout protection** - 2-minute max per inference
- **No shell execution** - Direct subprocess calls only
- **Audit logging** - All operations logged with timestamps

## How It Works

1. **Detection Phase** - Analyzes text for suspicious legal claims using prompt engineering
2. **Verification Phase** - Validates flagged claims through logical reasoning
3. **JSON Output** - Returns structured results with explanations


## Disclaimer

For educational and research purposes. Not a substitute for professional legal review. AI models can make mistakes—always verify critical legal claims independently.

