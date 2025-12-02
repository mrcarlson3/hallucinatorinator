"""
Legal Hallucination Detector
Three-stage pipeline: LLM Analysis, RAG Verification, Final Decision
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import ast
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

MODEL = "llama3:8b"
TIMEOUT = 180

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_ollama(prompt: str) -> str:
    """Execute Ollama model via subprocess."""
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


def extract_json(text: str) -> Dict:
    """Parse JSON from LLM response with comprehensive error handling."""
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    candidate = match.group(1).strip() if match else text
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    try:
        start = candidate.index('{')
        depth = 0
        end = start
        for i in range(start, len(candidate)):
            if candidate[i] == '{':
                depth += 1
            elif candidate[i] == '}':
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        candidate = candidate[start:end]
    except ValueError:
        pass
    
    fixed = candidate
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    fixed = re.sub(r':\s*True\b', ': true', fixed)
    fixed = re.sub(r':\s*False\b', ': false', fixed)
    fixed = re.sub(r':\s*None\b', ': null', fixed)
    
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    try:
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        pass
    
    raise ValueError(f"JSON parse failed: {candidate[:200]}")


def extract_citations(text: str) -> List[Dict]:
    """Extract legal citations from output text."""
    patterns = [
        (r'(\d{1,3})\s+(U\.?S\.?)\s+(\d{1,4})', "case"),
        (r'(\d{1,3})\s+(F\.?\s*(?:2d|3d|4th)?)\s+(\d{1,4})', "case"),
        (r'(\d{1,3})\s+(S\.?\s*Ct\.?)\s+(\d{1,4})', "case"),
        (r'([A-Z][a-zA-Z\-]+)\s+v\.?\s+([A-Z][a-zA-Z\-]+)', "case_name"),
        (r'(\d+)\s+(U\.?S\.?C\.?)\s*[§]+\s*(\d+)', "statute"),
    ]
    
    citations = []
    seen = set()
    
    for pattern, ctype in patterns:
        for match in re.finditer(pattern, text):
            content = match.group(0).strip()
            normalized = re.sub(r'\s+', ' ', content.lower())
            
            if normalized not in seen and len(content) > 5:
                seen.add(normalized)
                citations.append({
                    "type": ctype,
                    "content": content,
                    "position": match.start()
                })
    
    citations.sort(key=lambda x: x["position"])
    return citations


def stage1_llm_analysis(text: str, citations: List[Dict]) -> Dict:
    """
    Stage 1: Initial LLM analysis for potential hallucinations.
    The model examines the output for factual inconsistencies,
    suspicious citations, and logical coherence issues.
    """
    logger.info("Stage 1: Conducting initial LLM analysis")
    
    citation_summary = json.dumps(citations[:15], indent=2)
    
    prompt = f"""You are a legal document analyst. Your task is to examine the following text for potential hallucinations, which include fabricated citations, false statements of law, misquoted holdings, or invented case names.

DOCUMENT TEXT:
{text[:8000]}

EXTRACTED CITATIONS:
{citation_summary}

Analyze this document carefully. Consider the following:

1. Do the citations follow standard legal citation format?
2. Are there any citations that seem unusual or potentially fabricated?
3. Do the legal claims appear consistent with established law?
4. Are there any logical contradictions or implausible statements?
5. Do quoted passages seem authentic or potentially invented?

Provide your analysis as a JSON object with exactly this structure:
{{
    "is_hallucination": false,
    "confidence": 75,
    "findings": [
        {{"issue": "description of concern", "severity": "low", "location": "where in text"}}
    ],
    "suspicious_citations": ["list any citations that seem questionable"],
    "logic_coherent": true,
    "factual_concerns": ["list any factual claims that seem dubious"],
    "summary": "One paragraph assessment of document reliability"
}}

Return only valid JSON. Use double quotes for strings. Do not include trailing commas."""

    try:
        response = run_ollama(prompt)
        result = extract_json(response)
        
        result.setdefault("is_hallucination", False)
        result.setdefault("confidence", 50)
        result.setdefault("findings", [])
        result.setdefault("suspicious_citations", [])
        result.setdefault("logic_coherent", True)
        result.setdefault("factual_concerns", [])
        result.setdefault("summary", "Analysis completed")
        
        return result
        
    except Exception as e:
        logger.error(f"Stage 1 analysis failed: {e}")
        return {
            "is_hallucination": None,
            "confidence": 0,
            "findings": [],
            "suspicious_citations": [],
            "logic_coherent": None,
            "factual_concerns": [],
            "summary": "Analysis could not be completed",
            "error": str(e)
        }


def stage2_rag_verification(citations: List[Dict], llm_findings: Dict) -> Dict:
    """
    Stage 2: RAG verification against CourtListener database.
    Each citation is checked against the legal database to confirm existence.
    """
    logger.info("Stage 2: Verifying citations against CourtListener")
    
    verified = []
    unverified = []
    rag_context = ""
    
    try:
        from legal_rag import enhanced_scan_with_rag
        result = enhanced_scan_with_rag("", citations)
        verified = result.get("verified", [])
        unverified = result.get("unverified", [])
        rag_context = result.get("rag_context", "")
        verification_rate = result.get("verification_rate", 0)
        
    except ImportError:
        logger.warning("RAG module not available, skipping database verification")
        unverified = citations
        rag_context = "Database verification unavailable"
        verification_rate = 0
        
    except Exception as e:
        logger.error(f"RAG verification failed: {e}")
        unverified = citations
        rag_context = f"Verification error: {e}"
        verification_rate = 0
    
    # Identify conflicts between LLM suspicions and RAG results
    suspicious = llm_findings.get("suspicious_citations", [])
    conflicts = []
    false_positives = []
    
    for s in suspicious:
        s_lower = str(s).lower()
        found_in_verified = any(s_lower in str(v.get("content", "")).lower() for v in verified)
        if found_in_verified:
            conflicts.append({
                "citation": s,
                "status": "LLM flagged but RAG verified",
                "resolution": "Trust RAG verification"
            })
            false_positives.append(s)
    
    return {
        "verified": verified,
        "unverified": unverified,
        "rag_context": rag_context,
        "verification_rate": verification_rate,
        "conflicts": conflicts,
        "false_positives": false_positives,
        "total_citations": len(verified) + len(unverified)
    }


def stage3_final_decision(text: str, llm_analysis: Dict, rag_results: Dict) -> Dict:
    """
    Stage 3: Final determination combining LLM analysis and RAG verification.
    RAG results take precedence for citation verification as they represent
    ground truth from the legal database.
    """
    logger.info("Stage 3: Rendering final decision")
    
    verified_count = len(rag_results.get("verified", []))
    unverified_count = len(rag_results.get("unverified", []))
    total_citations = rag_results.get("total_citations", 0)
    verification_rate = rag_results.get("verification_rate", 0)
    
    # If RAG found unverified citations, this is strong evidence of hallucination
    if unverified_count > 0:
        unverified_list = [u.get("content", "") for u in rag_results.get("unverified", [])]
        
        return {
            "is_hallucination": True,
            "confidence": min(95, 70 + (unverified_count * 5)),
            "risk_level": "high" if unverified_count > 1 else "medium",
            "llm_weight": "30%",
            "rag_weight": "70%",
            "reasoning": (
                f"Database verification identified {unverified_count} citation(s) that could not be "
                f"confirmed in the CourtListener legal database. Unverified citations: "
                f"{', '.join(unverified_list[:5])}. This indicates potential fabrication of legal authorities."
            ),
            "confirmed_issues": unverified_list,
            "verified_citations": verified_count,
            "unverified_citations": unverified_count,
            "recommendations": [
                "Manually verify all unconfirmed citations before relying on this document",
                "Cross-reference with official court records",
                "Consider the document unreliable for legal proceedings"
            ],
            "safe_to_use": False
        }
    
    # All citations verified successfully
    if verified_count > 0 and unverified_count == 0:
        llm_flagged = llm_analysis.get("is_hallucination", False)
        llm_concerns = llm_analysis.get("findings", [])
        
        if llm_flagged and llm_concerns:
            # LLM found issues but citations checked out
            return {
                "is_hallucination": False,
                "confidence": 75,
                "risk_level": "low",
                "llm_weight": "40%",
                "rag_weight": "60%",
                "reasoning": (
                    f"All {verified_count} citation(s) were successfully verified in the legal database. "
                    f"While the initial analysis flagged potential concerns, database verification confirms "
                    f"the cited authorities exist. The output appears to reference legitimate legal sources."
                ),
                "confirmed_issues": [],
                "verified_citations": verified_count,
                "unverified_citations": 0,
                "recommendations": [
                    "Citations verified but consider reviewing flagged concerns",
                    "output citations appear legitimate"
                ],
                "safe_to_use": True,
                "llm_concerns_noted": [f.get("issue", "") for f in llm_concerns[:3]]
            }
        
        return {
            "is_hallucination": False,
            "confidence": 90,
            "risk_level": "low",
            "llm_weight": "30%",
            "rag_weight": "70%",
            "reasoning": (
                f"All {verified_count} extracted citation(s) were successfully verified against the "
                f"CourtListener database. No indicators of hallucination were detected. The output "
                f"appears to reference legitimate legal authorities."
            ),
            "confirmed_issues": [],
            "verified_citations": verified_count,
            "unverified_citations": 0,
            "recommendations": [
                "output citations appear legitimate",
                "Standard due diligence recommended before legal reliance"
            ],
            "safe_to_use": True
        }
    
    # No citations to verify, rely on LLM analysis
    llm_result = llm_analysis.get("is_hallucination", False)
    llm_confidence = llm_analysis.get("confidence", 50)
    
    return {
        "is_hallucination": llm_result,
        "confidence": max(llm_confidence - 20, 30),
        "risk_level": "medium",
        "llm_weight": "90%",
        "rag_weight": "10%",
        "reasoning": (
            "No legal citations were available for database verification. Assessment relies primarily "
            f"on textual analysis. {llm_analysis.get('summary', 'Review recommended.')}"
        ),
        "confirmed_issues": llm_analysis.get("factual_concerns", []),
        "verified_citations": 0,
        "unverified_citations": 0,
        "recommendations": [
            "Limited verification possible without citations",
            "Manual review recommended",
            "Consider additional fact-checking"
        ],
        "safe_to_use": not llm_result
    }


def generate_report(results: Dict) -> str:
    """Generate a professional narrative report from analysis results."""
    
    final = results.get("stage3_final", {})
    stage1 = results.get("stage1_llm", {})
    stage2 = results.get("stage2_rag", {})
    
    is_hallucination = final.get("is_hallucination", False)
    confidence = final.get("confidence", 0)
    risk_level = final.get("risk_level", "unknown")
    
    report_lines = [
        "LEGAL output ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Analysis Date: {results.get('timestamp', 'Unknown')}",
        f"Processing Time: {results.get('processing_time', 0)} seconds",
        f"Citations Analyzed: {results.get('citations_found', 0)}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        ""
    ]
    
    if is_hallucination:
        report_lines.append(
            f"This output has been flagged as containing potential hallucinations with "
            f"{confidence}% confidence. The risk level is assessed as {risk_level.upper()}. "
            f"Database verification was unable to confirm one or more cited legal authorities, "
            f"which suggests the output may contain fabricated or inaccurate citations."
        )
    else:
        report_lines.append(
            f"This output passed hallucination screening with {confidence}% confidence. "
            f"The risk level is assessed as {risk_level.upper()}. "
            f"Cited legal authorities were successfully verified against the CourtListener database."
        )
    
    report_lines.extend([
        "",
        "",
        "DETAILED FINDINGS",
        "-" * 40,
        "",
        final.get("reasoning", "No detailed reasoning available."),
        ""
    ])
    
    verified = stage2.get("verified", [])
    unverified = stage2.get("unverified", [])
    
    if verified:
        report_lines.extend([
            "",
            "Verified Citations:",
        ])
        for v in verified[:10]:
            report_lines.append(
                f"  - {v.get('content', 'Unknown')} confirmed as {v.get('case_name', 'Unknown case')}"
            )
    
    if unverified:
        report_lines.extend([
            "",
            "Unverified Citations (potential hallucinations):",
        ])
        for u in unverified[:10]:
            report_lines.append(f"  - {u.get('content', 'Unknown')} could not be verified")
    
    recommendations = final.get("recommendations", [])
    if recommendations:
        report_lines.extend([
            "",
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"  {i}. {rec}")
    
    report_lines.extend([
        "",
        "",
        "CONCLUSION",
        "-" * 40,
        ""
    ])
    
    safe = final.get("safe_to_use", False)
    if safe:
        report_lines.append(
            "Based on the analysis performed, this output may be considered for use with "
            "standard due diligence. The cited authorities appear to reference legitimate legal sources."
        )
    else:
        report_lines.append(
            "Based on the analysis performed, this output should not be relied upon without "
            "thorough manual verification. The identified issues suggest potential fabrication "
            "of legal authorities that could undermine any legal arguments based on this output."
        )
    
    report_lines.extend([
        "",
        "=" * 60,
        "END OF REPORT"
    ])
    
    return "\n".join(report_lines)


def analyze(text: str) -> Dict[str, Any]:
    """
    Main analysis pipeline.
    Executes three-stage hallucination detection: LLM analysis,
    RAG verification, and final decision synthesis.
    """
    start = time.time()
    
    citations = extract_citations(text)
    logger.info(f"Extracted {len(citations)} citations from output")
    
    stage1 = stage1_llm_analysis(text, citations)
    stage2 = stage2_rag_verification(citations, stage1)
    stage3 = stage3_final_decision(text, stage1, stage2)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "processing_time": round(time.time() - start, 2),
        "citations_found": len(citations),
        "extracted_citations": citations,
        "stage1_llm": stage1,
        "stage2_rag": stage2,
        "stage3_final": stage3,
        "result": {
            "is_hallucination": stage3.get("is_hallucination"),
            "confidence": stage3.get("confidence"),
            "risk_level": stage3.get("risk_level"),
            "safe_to_use": stage3.get("safe_to_use"),
            "reasoning": stage3.get("reasoning")
        }
    }
    
    results["narrative_report"] = generate_report(results)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if input_path.exists():
            text = input_path.read_text()
        else:
            print(f"File not found: {input_path}")
            sys.exit(1)
    else:
        print("Legal Hallucination Detector")
        print("Paste output text below, then press Ctrl+D (Unix) or Ctrl+Z (Windows) to analyze:")
        print("-" * 60)
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input provided")
        sys.exit(1)
    
    print("\nAnalyzing output...")
    print("-" * 60)
    
    result = analyze(text)
    
    print("\n")
    print(result["narrative_report"])
    
    output_file = Path(f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file.write_text(json.dumps(result, indent=2))
    print(f"\nDetailed results saved to: {output_file}")