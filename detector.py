"""
Legal Hallucination Detector - Secure local LLM-based verification
Detects fabricated cases, fake statutes, and unsupported legal claims.
"""

import json
import logging
import re
import subprocess
import time
from collections import deque
from typing import Any, Dict, List

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL = "llama3:8b"
MAX_INPUT_LENGTH = 50000  # 50KB limit
MAX_OUTPUT_LENGTH = 100000  # 100KB limit
TIMEOUT_SECONDS = 120  # 2 minute timeout
MAX_REQUESTS_PER_MINUTE = 10

# Rate limiting state
_request_timestamps = deque(maxlen=MAX_REQUESTS_PER_MINUTE)


def _check_rate_limit() -> None:
    """Enforce rate limiting to prevent abuse."""
    now = time.time()
    
    # Remove old timestamps
    while _request_timestamps and now - _request_timestamps[0] > 60:
        _request_timestamps.popleft()
    
    if len(_request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        logger.warning("Rate limit exceeded")
        raise RuntimeError(f"Rate limit exceeded: max {MAX_REQUESTS_PER_MINUTE} requests/minute")
    
    _request_timestamps.append(now)


def _sanitize_input(text: str) -> str:
    """Sanitize and validate input text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if len(text) > MAX_INPUT_LENGTH:
        logger.warning(f"Input truncated: {len(text)} -> {MAX_INPUT_LENGTH} chars")
        return text[:MAX_INPUT_LENGTH]
    
    return text


def _extract_json(text: str, expect_type: str = "object") -> str:
    """
    Extract JSON from model output, handling various formats.
    
    Args:
        text: Raw model output
        expect_type: "object" for {} or "array" for []
    
    Returns:
        Extracted JSON string
    """
    # Try to find JSON markers
    if expect_type == "array":
        start_char, end_char = "[", "]"
    else:
        start_char, end_char = "{", "}"
    
    try:
        start_idx = text.index(start_char)
        # Find matching closing bracket
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]
        
        # If no matching bracket found, try simple approach
        return text[start_idx:]
    except ValueError:
        raise ValueError(f"No {expect_type} found in output")


def _sanitize_output(text: str) -> str:
    """Sanitize output by removing control characters and enforcing length limits."""
    # Remove control characters (keep newline and tab)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    if len(text) > MAX_OUTPUT_LENGTH:
        logger.warning(f"Output truncated: {len(text)} -> {MAX_OUTPUT_LENGTH} chars")
        return text[:MAX_OUTPUT_LENGTH]
    
    return text


def _run_ollama(prompt: str) -> str:
    """
    Execute Ollama model with security protections.
    
    Security features:
    - Rate limiting
    - Input/output sanitization
    - Timeout protection
    - No shell execution
    - Error message sanitization
    """
    _check_rate_limit()
    prompt = _sanitize_input(prompt)
    
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Ollama failed with exit code {result.returncode}")
            raise RuntimeError("Model inference failed")
        
        output = result.stdout.decode("utf-8", errors="ignore").strip()
        return _sanitize_output(output)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Inference timeout after {TIMEOUT_SECONDS}s")
        raise RuntimeError("Model inference timeout")
    except FileNotFoundError:
        logger.error("Ollama not found - is it installed?")
        raise RuntimeError("Ollama not found. Install with: brew install ollama")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}")
        raise RuntimeError("Model inference failed")


def detect_hallucinations(text: str) -> List[Dict[str, Any]]:
    """
    Detect fabricated or unsupported legal claims in text.
    
    Returns:
        List of detected issues with explanations.
    """
    logger.info(f"Analyzing {len(text)} characters")
    text = _sanitize_input(text)

    prompt = """You are an expert U.S. legal claim evaluator with advanced reasoning skills.  
Your task is to determine whether the following legal claim is factually accurate, fabricated, or cannot be verified based on established U.S. law, federal and state court precedent, and recognized legal principles.

Your analysis must:
- Apply U.S. legal doctrines, statutory interpretation, and case-law reasoning.
- Consider jurisdiction, procedural posture, and relevant legal context.
- Identify whether the claim aligns with known legal outcomes, typical judicial reasoning, or established precedent.
- Distinguish verifiable facts from assertions that appear invented, implausible, or unsupported.
- Provide a concise but thorough legal explanation for your conclusion.

CRITICAL RULES:
1. Output ONLY a JSON array.  
2. Do NOT include any text outside the JSON.  
3. The JSON must contain objects with this format:

{"claim": "text", "why": "reasoning based on U.S. law"}

Claim: """ + text + """

JSON array:
"""

    try:
        output = _run_ollama(prompt)
        json_str = _extract_json(output, "array")
        result = json.loads(json_str)
        
        if not isinstance(result, list):
            raise ValueError("Expected JSON array")
        
        logger.info(f"Found {len(result)} potential issues")
        return result
        
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON parsing failed: {e}")
        return [{"error": "Invalid model output"}]
    except Exception as e:
        logger.error(f"Detection failed: {type(e).__name__}")
        return [{"error": "Detection failed"}]


def verify_claim(claim: str) -> Dict[str, Any]:
    """
    Verify a legal claim using logical reasoning.
    
    Returns:
        Dictionary with verdict and explanation.
    """
    logger.info(f"Verifying: {claim[:50]}...")
    claim = _sanitize_input(claim)
    
    prompt = """You are a legal claim verifier. Analyze this claim and determine if it's fabricated.

CRITICAL: Return ONLY a JSON object. No explanations, no extra text.

Format:
{"verdict": "Supported", "explanation": "reason"}
OR
{"verdict": "Fabricated", "explanation": "reason"}
OR
{"verdict": "Unknown", "explanation": "reason"}

Claim: """ + claim + """

JSON object:"""

    try:
        output = _run_ollama(prompt)
        json_str = _extract_json(output, "object")
        result = json.loads(json_str)
        
        if not isinstance(result, dict) or "verdict" not in result:
            raise ValueError("Invalid result structure")
        
        # Validate verdict
        valid_verdicts = {"Supported", "Fabricated", "Unknown"}
        if result["verdict"] not in valid_verdicts:
            logger.warning(f"Invalid verdict '{result['verdict']}', defaulting to Unknown")
            result["verdict"] = "Unknown"
        
        logger.info(f"Verdict: {result['verdict']}")
        return result
        
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON parsing failed: {e}")
        return {"verdict": "Unknown", "explanation": "Parsing failed"}
    except Exception as e:
        logger.error(f"Verification failed: {type(e).__name__}")
        return {"verdict": "Unknown", "explanation": "Verification failed"}


def main():
    """Main entry point for the detector."""
    print("=" * 70)
    print("Legal Hallucination Detector")
    print("=" * 70)
    print("\nEnter legal text to analyze (press Ctrl+D or Ctrl+Z when done):")
    print("-" * 70)
    
    # Read multiline input
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    text = "\n".join(lines).strip()
    
    if not text:
        print("\nNo text provided. Exiting.")
        return
    
    print("\n" + "=" * 70)
    logger.info("Starting hallucination detection")
    print("Analyzing text for hallucinations...\n")
    
    # Detect hallucinations
    results = detect_hallucinations(text)
    print(json.dumps(results, indent=2))
    
    # Verify each detected claim
    if results and not any("error" in r for r in results):
        print("\n" + "-" * 70)
        print("Verifying detected claims...")
        print("-" * 70)
        
        for item in results:
            claim = item.get("claim")
            if claim:
                print(f"\nClaim: {claim[:60]}...")
                verdict = verify_claim(claim)
                print(json.dumps(verdict, indent=2))
    
    print("\n" + "=" * 70)
    logger.info("Detection completed successfully")
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nProcess interrupted")
        exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {type(e).__name__} - {e}")
        print(f"\nError: {e}")
        exit(1)