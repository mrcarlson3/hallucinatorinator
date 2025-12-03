"""
Legal Hallucination Detector
Three-stage analysis: Initial assessment, RAG verification, Final synthesis
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODEL = "llama3:8b"
TIMEOUT = 180


def run_ollama(prompt: str) -> str:
    """Execute Ollama model."""
    result = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return result.stdout.decode().strip()


def extract_json(text: str) -> dict:
    """Parse JSON from LLM response."""
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        candidate = match.group(0)
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        candidate = re.sub(r':\s*True\b', ': true', candidate)
        candidate = re.sub(r':\s*False\b', ': false', candidate)
        return json.loads(candidate)
    raise ValueError("No JSON found in response")


def extract_citations(text: str) -> list:
    """Extract legal citations for RAG verification."""
    citations = []
    
    # Case citations
    for match in re.finditer(r'(\d{1,3})\s+(U\.?S\.?|F\.?\s*(?:2d|3d|4th)?|S\.?\s*Ct\.?)\s+(\d{1,4})', text):
        citations.append({"type": "case", "citation": match.group(0)})
    
    # Rule citations
    for match in re.finditer(r'(?:Fed\.?\s*R\.?\s*Civ\.?\s*P\.?|Rule|FRCP)\s+\d+(?:\s*\([a-zA-Z0-9]+\))*', text, re.IGNORECASE):
        citations.append({"type": "rule", "citation": match.group(0)})
    
    # Statutory citations
    for match in re.finditer(r'\d+\s+U\.?S\.?C\.?\s*§+\s*\d+', text):
        citations.append({"type": "statute", "citation": match.group(0)})
    
    return citations


def stage1_initial_analysis(text: str) -> dict:
    """Stage 1: Initial LLM assessment without external verification."""
    
    prompt = f"""You are a distinguished legal scholar with expertise in federal civil procedure, evidence, and statutory interpretation. You have decades of experience analyzing legal documents for accuracy.

Analyze the following legal text for potential errors or hallucinations:

---
{text[:10000]}
---

Examine this document for:
1. Nonexistent rule subdivisions (e.g., citing "Rule 12(a)(1)(F)(ii)" when that subdivision does not exist)
2. Incorrect statements of law or procedure
3. Fabricated exceptions or provisions
4. Wrong deadlines or time periods
5. Misattributed holdings or provisions

Provide your scholarly assessment as JSON:
{{
    "contains_errors": true or false,
    "errors_identified": [
        {{"issue": "description", "severity": "high/medium/low"}}
    ],
    "accurate_elements": ["list of elements that appear correct"],
    "preliminary_assessment": "Your professional narrative assessment"
}}

Return only valid JSON."""

    try:
        response = run_ollama(prompt)
        return extract_json(response)
    except Exception as e:
        return {"error": str(e), "contains_errors": None}


def stage2_rag_verification(text: str, citations: list, stage1: dict) -> dict:
    """Stage 2: RAG-enhanced verification of citations and claims."""
    
    rag_available = False
    rag_results = None
    
    try:
        from legal_rag import enhanced_scan_with_rag
        rag_available = True
        rag_results = enhanced_scan_with_rag(text, citations[:20])
    except ImportError:
        pass
    except Exception as e:
        rag_results = {"error": str(e)}
    
    # Second LLM pass with RAG context
    rag_context = ""
    if rag_results and "verified" in rag_results:
        rag_context = f"\n\nRAG VERIFICATION RESULTS:\nVerified citations: {rag_results.get('verified', [])}\nUnverified citations: {rag_results.get('unverified', [])}"
    
    errors_from_stage1 = stage1.get("errors_identified", [])
    
    prompt = f"""You are a distinguished legal scholar conducting a second-pass verification of a legal document.

ORIGINAL DOCUMENT:
---
{text[:8000]}
---

INITIAL ANALYSIS FINDINGS:
{json.dumps(errors_from_stage1, indent=2)}
{rag_context}

As a legal scholar, verify the initial findings. For each identified error, confirm whether it is truly an error or a false positive. Consider:
- Do the cited rule subdivisions actually exist in the Federal Rules?
- Are the stated legal principles accurate?
- Are deadlines and time periods correctly stated?

Provide your verification as JSON:
{{
    "confirmed_errors": [
        {{"issue": "description", "explanation": "why this is confirmed as an error"}}
    ],
    "false_positives": [
        {{"issue": "description", "explanation": "why initial assessment was wrong"}}
    ],
    "additional_concerns": ["any new issues discovered"],
    "verification_notes": "Your scholarly notes on the verification process"
}}

Return only valid JSON."""

    try:
        response = run_ollama(prompt)
        result = extract_json(response)
        result["rag_available"] = rag_available
        result["rag_results"] = rag_results
        return result
    except Exception as e:
        return {"error": str(e), "rag_available": rag_available}


def stage3_final_synthesis(text: str, stage1: dict, stage2: dict) -> dict:
    """Stage 3: Final synthesis and professional report."""
    
    confirmed_errors = stage2.get("confirmed_errors", [])
    stage1_errors = stage1.get("errors_identified", [])
    
    # If stage 2 didn't run properly, use stage 1 findings
    if not confirmed_errors and stage1_errors:
        confirmed_errors = [{"issue": e.get("issue", ""), "explanation": "From initial analysis"} 
                          for e in stage1_errors if e.get("severity") == "high"]
    
    error_count = len(confirmed_errors)
    
    prompt = f"""You are a distinguished legal scholar preparing a final assessment of a legal document's reliability.

DOCUMENT EXCERPT:
---
{text[:6000]}
---

CONFIRMED ERRORS:
{json.dumps(confirmed_errors, indent=2)}

INITIAL ASSESSMENT:
{stage1.get("preliminary_assessment", "No initial assessment available.")}

VERIFICATION NOTES:
{stage2.get("verification_notes", "No verification notes available.")}

Prepare your final scholarly assessment as JSON:
{{
    "narrative_assessment": "A professional 2-3 paragraph assessment of this document's reliability, written as a legal scholar would write for a colleague. Discuss what was analyzed, what errors were found (if any), and the implications.",
    "confidence_score": 0-100,
    "risk_level": "low/medium/high/critical",
    "recommendation": "clear recommendation on whether this document can be relied upon"
}}

The confidence score reflects your confidence in YOUR assessment (not the document's accuracy).
Risk level reflects the danger of relying on this document if it contains the identified errors.

Return only valid JSON."""

    try:
        response = run_ollama(prompt)
        result = extract_json(response)
        result["error_count"] = error_count
        result["confirmed_errors"] = confirmed_errors
        return result
    except Exception as e:
        return {
            "error": str(e),
            "error_count": error_count,
            "confirmed_errors": confirmed_errors,
            "narrative_assessment": "Final synthesis could not be completed.",
            "confidence_score": 0,
            "risk_level": "unknown"
        }


def format_report(stage1: dict, stage2: dict, stage3: dict, processing_time: float) -> str:
    """Format the final report."""
    
    lines = [
        "",
        "═" * 70,
        "LEGAL DOCUMENT ACCURACY ANALYSIS",
        "═" * 70,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Processing Time: {processing_time:.1f} seconds",
        "",
        "─" * 70,
        "STAGE 1: INITIAL SCHOLARLY ASSESSMENT",
        "─" * 70,
        "",
        stage1.get("preliminary_assessment", "No assessment available."),
        ""
    ]
    
    if stage1.get("errors_identified"):
        lines.append("Potential issues identified:")
        for err in stage1["errors_identified"]:
            lines.append(f"  • {err.get('issue', 'Unknown')} [{err.get('severity', 'unknown')}]")
        lines.append("")
    
    lines.extend([
        "─" * 70,
        "STAGE 2: RAG-ENHANCED VERIFICATION",
        "─" * 70,
        "",
        f"RAG Database Available: {'Yes' if stage2.get('rag_available') else 'No'}",
        "",
        stage2.get("verification_notes", "No verification notes available."),
        ""
    ])
    
    if stage2.get("confirmed_errors"):
        lines.append("Confirmed errors:")
        for err in stage2["confirmed_errors"]:
            lines.append(f"  • {err.get('issue', 'Unknown')}")
            lines.append(f"    → {err.get('explanation', '')}")
        lines.append("")
    
    if stage2.get("false_positives"):
        lines.append("False positives (initially flagged but verified as correct):")
        for fp in stage2["false_positives"]:
            lines.append(f"  • {fp.get('issue', 'Unknown')}")
        lines.append("")
    
    lines.extend([
        "─" * 70,
        "STAGE 3: FINAL ASSESSMENT",
        "─" * 70,
        "",
        stage3.get("narrative_assessment", "No final assessment available."),
        "",
        "─" * 70,
        "",
        f"CONFIDENCE SCORE: {stage3.get('confidence_score', 'N/A')}%",
        f"RISK LEVEL: {stage3.get('risk_level', 'N/A').upper()}",
        "",
        f"RECOMMENDATION: {stage3.get('recommendation', 'No recommendation available.')}",
        "",
        "═" * 70,
    ])
    
    return "\n".join(lines)


def analyze(text: str) -> str:
    """Main analysis pipeline."""
    start = time.time()
    
    # Extract citations for RAG
    citations = extract_citations(text)
    
    # Three-stage analysis
    stage1 = stage1_initial_analysis(text)
    stage2 = stage2_rag_verification(text, citations, stage1)
    stage3 = stage3_final_synthesis(text, stage1, stage2)
    
    processing_time = time.time() - start
    
    return format_report(stage1, stage2, stage3, processing_time)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if input_path.exists():
            text = input_path.read_text()
        else:
            print(f"File not found: {input_path}")
            sys.exit(1)
    else:
        print("Legal Hallucination Detector")
        print("Paste document text below, then press Ctrl+D (Unix) or Ctrl+Z (Windows):")
        print("-" * 60)
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input provided")
        sys.exit(1)
    
    print("\nAnalyzing document...")
    report = analyze(text)
    print(report)