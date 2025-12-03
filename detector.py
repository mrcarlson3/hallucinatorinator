"""
Legal Hallucination Detector
Three-stage analysis using LegalRAG for citation verification
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

# Import the RAG system
from legal_rag import LegalRAG, enhanced_scan_with_rag

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


def extract_citations(text: str) -> list:
    """Extract legal citations from text."""
    citations = []
    
    # Full case citations: "Name v. Name, 123 F.2d 456 (Court Year)"
    case_pattern = r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+v\.?\s+([A-Z][a-zA-Z\'\-\s]+?),?\s+(\d{1,3})\s+(F\.?\s*(?:2d|3d|4th)?|U\.?S\.?|S\.?\s*Ct\.?)\s+(\d{1,4})(?:\s*\(([^)]+)\))?'
    
    for match in re.finditer(case_pattern, text):
        plaintiff = match.group(1).strip()
        defendant = match.group(2).strip().rstrip(',')
        volume = match.group(3)
        reporter = match.group(4).replace(" ", "")
        page = match.group(5)
        court_year = match.group(6) if match.group(6) else ""
        
        full_citation = f"{volume} {reporter} {page}"
        case_name = f"{plaintiff} v. {defendant}"
        
        citations.append({
            "type": "case",
            "content": full_citation,
            "case_name": case_name,
            "plaintiff": plaintiff,
            "defendant": defendant,
            "volume": volume,
            "reporter": reporter,
            "page": page,
            "court_year": court_year,
            "original_text": match.group(0)
        })
    
    # Standalone citations without case names: "123 F.2d 456"
    standalone_pattern = r'(?<![a-zA-Z])(\d{1,3})\s+(F\.?\s*(?:2d|3d|4th)?|U\.?S\.?|S\.?\s*Ct\.?)\s+(\d{1,4})(?!\d)'
    
    for match in re.finditer(standalone_pattern, text):
        volume = match.group(1)
        reporter = match.group(2).replace(" ", "")
        page = match.group(3)
        full_citation = f"{volume} {reporter} {page}"
        
        already_found = any(c.get("content") == full_citation for c in citations)
        if not already_found:
            citations.append({
                "type": "case",
                "content": full_citation,
                "case_name": None,
                "volume": volume,
                "reporter": reporter,
                "page": page,
                "original_text": match.group(0)
            })
    
    # Case names without full citations: "Name v. Name"
    name_pattern = r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+v\.?\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)'
    
    for match in re.finditer(name_pattern, text):
        case_name = f"{match.group(1).strip()} v. {match.group(2).strip()}"
        
        already_found = any(c.get("case_name") == case_name for c in citations)
        if not already_found:
            citations.append({
                "type": "case_name",
                "content": case_name,
                "case_name": case_name,
                "original_text": match.group(0)
            })
    
    return citations


def verify_citations_with_rag(citations: list) -> dict:
    """Use LegalRAG to verify all citations."""
    
    if not citations:
        return {
            "verified": [],
            "unverified": [],
            "rag_context": "No citations found to verify.",
            "total_checked": 0,
            "verification_rate": 0
        }
    
    print(f"     Found {len(citations)} citations to verify")
    
    results = enhanced_scan_with_rag("", citations)
    
    v_count = len(results.get("verified", []))
    u_count = len(results.get("unverified", []))
    print(f"     Results: {v_count} verified, {u_count} unverified")
    
    return results


def stage1_initial_analysis(text: str, rag_results: dict) -> dict:
    """Stage 1: Initial LLM assessment informed by RAG verification."""
    
    verified = rag_results.get("verified", [])
    unverified = rag_results.get("unverified", [])
    rag_context = rag_results.get("rag_context", "")
    
    if unverified:
        citation_status = f"""
WARNING: {len(unverified)} CITATION(S) COULD NOT BE VERIFIED IN COURTLISTENER:
"""
        for item in unverified:
            citation_status += f"  - {item.get('content', 'Unknown')}: {item.get('reason', 'Not found')}\n"
        
        if verified:
            citation_status += f"\n{len(verified)} citation(s) were verified.\n"
    elif verified:
        citation_status = f"\nAll {len(verified)} citations were verified in CourtListener.\n"
    else:
        citation_status = "\nNo case citations were found to verify.\n"
    
    prompt = f"""You are a legal scholar analyzing a document for potential hallucinations.

COURTLISTENER DATABASE VERIFICATION RESULTS:{citation_status}

VERIFICATION DETAILS:
{rag_context}

DOCUMENT TEXT:
---
{text[:7000]}
---

Based on the citation verification results above, provide your scholarly assessment:

1. If citations COULD NOT BE VERIFIED, this strongly suggests hallucination - fabricated cases
2. Analyze other claims for accuracy: legal standards, procedures, dates
3. Assess overall reliability

Write 2-3 paragraphs. Be direct about hallucination risks."""

    try:
        response = run_ollama(prompt)
        return {
            "preliminary_assessment": response,
            "unverified_count": len(unverified),
            "verified_count": len(verified)
        }
    except Exception as e:
        return {"error": str(e), "preliminary_assessment": f"Analysis failed: {e}"}


def stage2_hallucination_analysis(text: str, stage1: dict, rag_results: dict) -> dict:
    """Stage 2: Focused hallucination analysis."""
    
    unverified = rag_results.get("unverified", [])
    verified = rag_results.get("verified", [])
    
    if not unverified:
        return {
            "verification_notes": f"All {len(verified)} case citations were verified against CourtListener. The cited cases exist in legal databases.",
            "hallucination_detected": False
        }
    
    unverified_detail = "\n".join([
        f"  - {item.get('content', 'Unknown')}: {item.get('reason', 'Not found in database')}"
        for item in unverified
    ])
    
    prompt = f"""You are a legal scholar investigating CONFIRMED citation problems.

THE FOLLOWING CITATIONS FAILED VERIFICATION IN COURTLISTENER:
{unverified_detail}

These citations were searched in CourtListener (a comprehensive legal database with millions of cases) and NOT FOUND. This is strong evidence the cases are fabricated/hallucinated.

VERIFIED CITATIONS (for comparison):
{len(verified)} citations were successfully verified.

ORIGINAL DOCUMENT:
---
{text[:5000]}
---

Analyze:
1. What specific claims depend on these unverified citations?
2. If the cases don't exist, what does that mean for the document's reliability?
3. Are there any other red flags (wrong dates, impossible procedures, etc.)?

Write 2-3 paragraphs. Treat unverified citations as PROBABLE HALLUCINATIONS."""

    try:
        response = run_ollama(prompt)
        return {
            "verification_notes": response,
            "hallucination_detected": True,
            "unverified_citations": unverified
        }
    except Exception as e:
        return {"error": str(e), "verification_notes": f"Analysis failed: {e}"}


def stage3_final_synthesis(text: str, stage1: dict, stage2: dict, rag_results: dict) -> dict:
    """Stage 3: Final synthesis with mandatory risk assessment."""
    
    verified = rag_results.get("verified", [])
    unverified = rag_results.get("unverified", [])
    verification_rate = rag_results.get("verification_rate", 0)
    
    if unverified:
        if len(unverified) > len(verified):
            forced_risk = "critical"
            risk_reason = f"Majority of citations ({len(unverified)}/{len(verified)+len(unverified)}) could not be verified"
        else:
            forced_risk = "high"
            risk_reason = f"{len(unverified)} citation(s) could not be verified - probable hallucination"
    else:
        forced_risk = "low"
        risk_reason = "All citations verified in CourtListener"
    
    prompt = f"""You are a legal scholar writing a final assessment.

VERIFICATION SUMMARY:
- Verified citations: {len(verified)}
- Unverified citations: {len(unverified)}
- Verification rate: {verification_rate:.1f}%
- Risk assessment: {risk_reason}

STAGE 1 FINDINGS:
{stage1.get('preliminary_assessment', 'N/A')[:1200]}

STAGE 2 FINDINGS:
{stage2.get('verification_notes', 'N/A')[:1200]}

Write a final 2-3 paragraph professional assessment summarizing:
1. What was analyzed
2. Whether hallucinations were detected (be specific about which citations)
3. Reliability conclusion

End with exactly these three lines:
CONFIDENCE: [number 0-100]
RISK: [low/medium/high/critical]
RECOMMENDATION: [one clear sentence]"""

    try:
        response = run_ollama(prompt)
        
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.IGNORECASE)
        risk_match = re.search(r'RISK:\s*(low|medium|high|critical)', response, re.IGNORECASE)
        rec_match = re.search(r'RECOMMENDATION:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        
        llm_risk = risk_match.group(1).lower() if risk_match else "unknown"
        final_risk = forced_risk if unverified else llm_risk
        
        llm_confidence = int(conf_match.group(1)) if conf_match else 50
        if unverified:
            max_confidence = int(100 * len(verified) / (len(verified) + len(unverified)))
            final_confidence = min(llm_confidence, max_confidence, 40)
        else:
            final_confidence = llm_confidence
        
        if unverified:
            recommendation = f"DO NOT RELY ON THIS DOCUMENT - {len(unverified)} citation(s) could not be verified in CourtListener and may be fabricated."
        else:
            recommendation = rec_match.group(1).strip() if rec_match else "Document appears reliable based on citation verification."
        
        return {
            "narrative_assessment": response,
            "confidence_score": final_confidence,
            "risk_level": final_risk,
            "recommendation": recommendation,
            "hallucination_detected": bool(unverified),
            "verification_rate": verification_rate
        }
    except Exception as e:
        return {
            "error": str(e),
            "narrative_assessment": f"Synthesis failed: {e}",
            "confidence_score": 20 if unverified else 50,
            "risk_level": forced_risk,
            "recommendation": "Analysis incomplete - verify all citations manually.",
            "hallucination_detected": bool(unverified)
        }


def format_report(stage1: dict, stage2: dict, stage3: dict, rag_results: dict, citations: list, processing_time: float) -> str:
    """Format the final report."""
    
    verified = rag_results.get("verified", [])
    unverified = rag_results.get("unverified", [])
    verification_rate = rag_results.get("verification_rate", 0)
    
    if unverified:
        header_status = "HALLUCINATION LIKELY DETECTED"
    else:
        header_status = "NO HALLUCINATIONS DETECTED"
    
    lines = [
        "",
        "=" * 70,
        "LEGAL HALLUCINATION DETECTION REPORT",
        header_status,
        "=" * 70,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Processing Time: {processing_time:.1f} seconds",
        f"Citations Found: {len(citations)}",
        f"Verification Rate: {verification_rate:.1f}%",
        "",
        "-" * 70,
        "CITATION VERIFICATION RESULTS (via CourtListener API)",
        "-" * 70,
        ""
    ]
    
    if unverified:
        lines.append(f"UNVERIFIED CITATIONS: {len(unverified)}")
        lines.append("")
        for item in unverified:
            lines.append(f"   - {item.get('content', 'Unknown')}")
            lines.append(f"     Status: {item.get('reason', 'Not found in database')}")
            lines.append("")
    
    if verified:
        lines.append(f"VERIFIED CITATIONS: {len(verified)}")
        lines.append("")
        for item in verified:
            lines.append(f"   - {item.get('content', 'Unknown')}")
            lines.append(f"     Case: {item.get('case_name', 'N/A')}")
            lines.append(f"     Court: {item.get('court', 'N/A')}")
            lines.append(f"     Date: {item.get('date_filed', 'N/A')}")
            lines.append(f"     Confidence: {item.get('confidence', 'N/A')}%")
            lines.append("")
    
    if not verified and not unverified:
        lines.append("No case citations found in document.")
        lines.append("")
    
    lines.extend([
        "-" * 70,
        "STAGE 1: INITIAL ASSESSMENT",
        "-" * 70,
        "",
        stage1.get("preliminary_assessment", "No assessment available."),
        "",
        "-" * 70,
        "STAGE 2: HALLUCINATION ANALYSIS",
        "-" * 70,
        "",
        stage2.get("verification_notes", "No analysis available."),
        "",
        "-" * 70,
        "STAGE 3: FINAL ASSESSMENT",
        "-" * 70,
        "",
        stage3.get("narrative_assessment", "No assessment available."),
        "",
        "=" * 70,
        "FINAL DETERMINATION",
        "=" * 70,
        ""
    ])
    
    risk = stage3.get("risk_level", "unknown").upper()
    
    lines.extend([
        f"CONFIDENCE: {stage3.get('confidence_score', 'N/A')}%",
        f"RISK LEVEL: {risk}",
        "",
        f"RECOMMENDATION: {stage3.get('recommendation', 'N/A')}",
        "",
        "=" * 70,
    ])
    
    return "\n".join(lines)


def analyze(text: str) -> str:
    """Main analysis pipeline."""
    start = time.time()
    
    print("\n" + "=" * 60)
    print("LEGAL HALLUCINATION DETECTOR")
    print("=" * 60)
    
    print("\n  Extracting citations...")
    citations = extract_citations(text)
    case_citations = [c for c in citations if c["type"] in ("case", "case_name")]
    print(f"     Found {len(case_citations)} case citations/names")
    
    for c in case_citations[:10]:
        if c.get("case_name"):
            print(f"       - {c.get('case_name')}: {c.get('content', c.get('original_text', ''))}")
        else:
            print(f"       - {c.get('content', c.get('original_text', ''))}")
    
    print("\n  Verifying citations with LegalRAG (CourtListener API)...")
    rag_results = verify_citations_with_rag(case_citations)
    
    print("\n  Stage 1: Initial analysis...")
    stage1 = stage1_initial_analysis(text, rag_results)
    
    print("  Stage 2: Hallucination analysis...")
    stage2 = stage2_hallucination_analysis(text, stage1, rag_results)
    
    print("  Stage 3: Final synthesis...")
    stage3 = stage3_final_synthesis(text, stage1, stage2, rag_results)
    
    processing_time = time.time() - start
    print(f"\n  Complete in {processing_time:.1f}s")
    
    return format_report(stage1, stage2, stage3, rag_results, citations, processing_time)


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
        print("Paste document text, then press Ctrl+D (Unix) or Ctrl+Z (Windows):")
        print("-" * 60)
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input provided")
        sys.exit(1)
    
    report = analyze(text)
    print(report)