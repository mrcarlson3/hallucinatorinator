"""
Legal Hallucination Detector - Secure local LLM-based verification for legal content
Designed for pro se litigants and lawyers to detect fabricated cases, fake statutes,
misquoted holdings, and unsupported legal claims in AI-generated legal content.
"""

import json
import logging
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configuration
MODEL = "llama3:8b"
MAX_INPUT_LENGTH = 50000
MAX_OUTPUT_LENGTH = 100000
TIMEOUT_SECONDS = 120
MAX_REQUESTS_PER_MINUTE = 10
LOG_FILE = Path("legal_hallucination_detector.log")

# Setup logging
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("LegalHallucinationDetector")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()
_request_timestamps = deque(maxlen=MAX_REQUESTS_PER_MINUTE)


def _check_rate_limit() -> None:
    now = time.time()
    while _request_timestamps and now - _request_timestamps[0] > 60:
        _request_timestamps.popleft()
    if len(_request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        raise RuntimeError(f"Rate limit exceeded: max {MAX_REQUESTS_PER_MINUTE} requests/minute")
    _request_timestamps.append(now)


def _sanitize_input(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return text[:MAX_INPUT_LENGTH] if len(text) > MAX_INPUT_LENGTH else text


def _sanitize_output(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text[:MAX_OUTPUT_LENGTH] if len(text) > MAX_OUTPUT_LENGTH else text


def _extract_json(text: str, expect_type: str = "object") -> str:
    start_char, end_char = ("[", "]") if expect_type == "array" else ("{", "}")
    try:
        start_idx = text.index(start_char)
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]
        return text[start_idx:]
    except ValueError:
        raise ValueError(f"No {expect_type} found in output")


def _run_ollama(prompt: str) -> str:
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
            raise RuntimeError("Model inference failed")
        
        output = result.stdout.decode("utf-8", errors="ignore").strip()
        return _sanitize_output(output)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Model inference timeout")
    except FileNotFoundError:
        raise RuntimeError("Ollama not found. Install with: brew install ollama")


def extract_legal_citations(text: str) -> List[Dict[str, str]]:
    """Extract legal citations from text."""
    extracted = []
    
    patterns = {
        "case_citation": [
            r'\d+\s+U\.?S\.?\s+\d+',
            r'\d+\s+S\.?\s*Ct\.?\s+\d+',
            r'\d+\s+F\.?\s*(?:2d|3d|4th)?\s+\d+',
            r'\d+\s+F\.?\s*Supp\.?\s*(?:2d|3d)?\s+\d+',
            r'\d{4}\s+WL\s+\d+',
        ],
        "case_name": [
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+v\.?\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
        ],
        "statute": [
            r'\d+\s+U\.?S\.?C\.?\s*§+\s*\d+(?:\([a-zA-Z0-9]+\))*',
            r'\d+\s+C\.?F\.?R\.?\s*§+\s*\d+(?:\.\d+)*',
            r'Fed\.?\s*R\.?\s*(?:Civ\.?|Crim\.?|App\.?|Evid\.?)?\s*P\.?\s*\d+',
        ],
    }
    
    for citation_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                content = " v. ".join(match) if isinstance(match, tuple) else match
                extracted.append({"type": citation_type, "content": content.strip()})
    
    # Deduplicate
    seen = set()
    return [e for e in extracted if e["content"].lower() not in seen and not seen.add(e["content"].lower())]


def scan_for_hallucinations(text: str) -> Dict[str, Any]:
    """Scan legal content for potential hallucinations."""
    text = _sanitize_input(text)
    extracted_citations = extract_legal_citations(text)
    
    prompt = """You are a legal fact-checker identifying fabricated content in AI-generated legal text.

Analyze for:
1. FABRICATED CASES - Case names/citations that don't exist
2. FAKE STATUTES - Incorrect statutory citations
3. MISQUOTED HOLDINGS - Inaccurate quotes attributed to cases
4. INCORRECT LEGAL PRINCIPLES - Wrongly stated legal rules
5. JURISDICTIONAL ERRORS - Wrong jurisdiction applied

Return ONLY this JSON:
{
    "issues": [
        {
            "text": "problematic text",
            "type": "fabricated_case|fake_statute|misquoted_holding|incorrect_principle|jurisdictional_error",
            "reasoning": "why this is flagged",
            "confidence": 0-100
        }
    ],
    "overall_risk": "low|medium|high|critical",
    "summary": "brief summary"
}

TEXT TO ANALYZE:
\"\"\"
""" + text + """
\"\"\"

JSON:"""

    try:
        output = _run_ollama(prompt)
        result = json.loads(_extract_json(output))
        result["extracted_citations"] = extracted_citations
        return result
    except (ValueError, json.JSONDecodeError):
        return {"error": "Analysis failed", "extracted_citations": extracted_citations}


def self_verify(original_text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the analysis and provide hallucination probability."""
    
    prompt = """Review this hallucination analysis of AI-generated legal content.

Return ONLY this JSON:
{
    "hallucination_probability": 0-100,
    "reasoning": "brief explanation",
    "missed_issues": ["any missed problems"],
    "court_filing_ready": true|false
}

ORIGINAL TEXT:
\"\"\"
""" + original_text[:5000] + """
\"\"\"

ANALYSIS:
\"\"\"
""" + json.dumps(analysis, indent=2)[:3000] + """
\"\"\"

JSON:"""

    try:
        output = _run_ollama(prompt)
        return json.loads(_extract_json(output))
    except (ValueError, json.JSONDecodeError):
        return {"hallucination_probability": 50, "court_filing_ready": False, "reasoning": "Verification failed"}


def analyze_legal_content(text: str) -> Dict[str, Any]:
    """Main analysis function."""
    logger.info(f"Analyzing {len(text)} characters")
    start_time = time.time()
    
    scan_result = scan_for_hallucinations(text)
    if "error" in scan_result:
        return scan_result
    
    verification = self_verify(text, scan_result)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "hallucination_probability": verification.get("hallucination_probability", 50),
        "court_filing_ready": verification.get("court_filing_ready", False),
        "overall_risk": scan_result.get("overall_risk", "unknown"),
        "summary": scan_result.get("summary", ""),
        "issues": scan_result.get("issues", []),
        "citations_found": scan_result.get("extracted_citations", []),
        "verification_reasoning": verification.get("reasoning", ""),
        "processing_seconds": round(time.time() - start_time, 2)
    }


def format_report(result: Dict[str, Any]) -> str:
    """Format analysis as a clean report."""
    lines = [
        "=" * 60,
        "LEGAL HALLUCINATION ANALYSIS",
        "=" * 60,
        "",
        f"Hallucination Probability: {result.get('hallucination_probability', 'N/A')}%",
        f"Overall Risk: {result.get('overall_risk', 'unknown').upper()}",
        f"Court Filing Ready: {'Yes' if result.get('court_filing_ready') else 'No'}",
        "",
        f"Summary: {result.get('summary', 'N/A')}",
        ""
    ]
    
    issues = result.get("issues", [])
    if issues:
        lines.append("-" * 60)
        lines.append(f"ISSUES FOUND ({len(issues)}):")
        lines.append("-" * 60)
        for i, issue in enumerate(sorted(issues, key=lambda x: x.get("confidence", 0), reverse=True), 1):
            lines.append(f"\n{i}. [{issue.get('type', 'unknown').upper()}] (Confidence: {issue.get('confidence', 'N/A')}%)")
            lines.append(f"   Text: {issue.get('text', 'N/A')[:100]}")
            lines.append(f"   Reason: {issue.get('reasoning', 'N/A')}")
    
    citations = result.get("citations_found", [])
    if citations:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"CITATIONS EXTRACTED ({len(citations)}):")
        lines.append("-" * 60)
        for c in citations[:10]:
            lines.append(f"  [{c.get('type', 'unknown')}] {c.get('content', 'N/A')}")
        if len(citations) > 10:
            lines.append(f"  ... and {len(citations) - 10} more")
    
    lines.extend([
        "",
        "=" * 60,
        "IMPORTANT: Verify all citations in Westlaw, LexisNexis, or",
        "Google Scholar before use in any court filing.",
        "=" * 60
    ])
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("LEGAL HALLUCINATION DETECTOR")
    print("=" * 60)
    print("\nPaste AI-generated legal text (Ctrl+D/Ctrl+Z to finish):\n")
    
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    
    text = "\n".join(lines).strip()
    if not text:
        print("\nNo text provided.")
        return
    
    print("\nAnalyzing...\n")
    result = analyze_legal_content(text)
    print(format_report(result))
    
    # Save JSON
    output_file = Path(f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\nError: {e}")