"""
Legal RAG - CourtListener API v4
Citation verification using CourtListener's REST API.
"""

import os
import requests
import hashlib
import json
import re
import logging
from pathlib import Path
from typing import List, Dict

COURTLISTENER_API = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_TOKEN = os.environ.get("COURTLISTENER_TOKEN", "")
CACHE_DIR = Path("legal_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LegalRAG:
    """CourtListener API v4 client for citation verification."""
    
    def __init__(self):
        self.headers = {"User-Agent": "LegalHallucinationDetector/1.0"}
        if COURTLISTENER_TOKEN:
            self.headers["Authorization"] = f"Token {COURTLISTENER_TOKEN}"
    
    def search_opinions(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the opinions index via the search endpoint."""
        try:
            resp = requests.get(
                f"{COURTLISTENER_API}/search/",
                params={"q": query, "type": "o", "page_size": max_results},
                headers=self.headers,
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
            logger.warning(f"Search returned status {resp.status_code}")
        except requests.RequestException as e:
            logger.error(f"Search request failed: {e}")
        return []
    
    def citation_lookup(self, citation: str) -> Dict:
        """Query the citation-lookup endpoint for exact citation matching."""
        try:
            resp = requests.get(
                f"{COURTLISTENER_API}/citation-lookup/",
                params={"citation": citation},
                headers=self.headers,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    return {"found": True, "data": results[0], "count": len(results)}
        except requests.RequestException as e:
            logger.error(f"Citation lookup failed: {e}")
        return {"found": False, "data": None, "count": 0}
    
    def get_cluster(self, cluster_id: int) -> Dict:
        """Retrieve opinion cluster details."""
        try:
            resp = requests.get(
                f"{COURTLISTENER_API}/clusters/{cluster_id}/",
                headers=self.headers,
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException as e:
            logger.error(f"Cluster lookup failed: {e}")
        return {}
    
    def verify_citation(self, citation: str) -> Dict:
        """
        Verify a legal citation exists in CourtListener.
        Uses multiple search strategies for thorough verification.
        """
        cache_key = hashlib.md5(citation.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            logger.info(f"Cache hit for: {citation}")
            return cached
        
        result = {
            "citation": citation,
            "verified": False,
            "source": "CourtListener",
            "case_name": None,
            "court": None,
            "date_filed": None,
            "docket_number": None,
            "search_method": None,
            "confidence": 0
        }
        
        # Strategy 1: Direct citation lookup
        lookup = self.citation_lookup(citation)
        if lookup["found"]:
            data = lookup["data"]
            result.update({
                "verified": True,
                "case_name": data.get("caseName") or data.get("case_name"),
                "court": data.get("court"),
                "date_filed": data.get("dateFiled") or data.get("date_filed"),
                "docket_number": data.get("docketNumber") or data.get("docket_number"),
                "search_method": "citation-lookup",
                "confidence": 90
            })
            cache_file.write_text(json.dumps(result, indent=2))
            return result
        
        # Strategy 2: Search by formatted citation
        search_results = self.search_opinions(f'citation:"{citation}"', max_results=5)
        if not search_results:
            search_results = self.search_opinions(citation, max_results=5)
        
        for r in search_results:
            if self._citation_matches(citation, r):
                result.update({
                    "verified": True,
                    "case_name": r.get("caseName") or r.get("case_name"),
                    "court": r.get("court"),
                    "date_filed": r.get("dateFiled") or r.get("date_filed"),
                    "docket_number": r.get("docketNumber") or r.get("docket_number"),
                    "search_method": "search",
                    "confidence": 80
                })
                break
        
        cache_file.write_text(json.dumps(result, indent=2))
        return result
    
    def _citation_matches(self, citation: str, result: Dict) -> bool:
        """Determine if a search result matches the given citation."""
        citation_lower = citation.lower()
        result_str = json.dumps(result).lower()
        
        # Federal reporter pattern: volume + reporter + page
        federal_match = re.search(
            r'(\d+)\s+(u\.?s\.?|f\.?\s*(?:2d|3d|4th)?|s\.?\s*ct\.?)\s+(\d+)',
            citation_lower
        )
        if federal_match:
            volume, reporter, page = federal_match.groups()
            if volume in result_str and page in result_str:
                return True
        
        # Party names pattern
        party_match = re.search(r'([a-z]+)\s+v\.?\s+([a-z]+)', citation_lower)
        if party_match:
            party1, party2 = party_match.groups()
            if len(party1) > 2 and len(party2) > 2:
                if party1 in result_str and party2 in result_str:
                    return True
        
        return False
    
    def verify_case_name(self, case_name: str) -> Dict:
        """Verify a case exists by party names."""
        cache_key = hashlib.md5(f"name:{case_name}".encode()).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        
        result = {
            "case_name": case_name,
            "verified": False,
            "source": "CourtListener",
            "official_citation": None,
            "court": None,
            "date_filed": None,
            "confidence": 0
        }
        
        search_results = self.search_opinions(case_name, max_results=10)
        
        for r in search_results:
            result_name = (r.get("caseName") or r.get("case_name") or "").lower()
            if self._names_match(case_name.lower(), result_name):
                result.update({
                    "verified": True,
                    "official_citation": r.get("citation", []),
                    "court": r.get("court"),
                    "date_filed": r.get("dateFiled") or r.get("date_filed"),
                    "confidence": 85
                })
                break
        
        cache_file.write_text(json.dumps(result, indent=2))
        return result
    
    def _names_match(self, query: str, result_name: str) -> bool:
        """Check if party names match between query and result."""
        match = re.search(r'([a-z]{3,})\s+v\.?\s+([a-z]{3,})', query)
        if match:
            party1, party2 = match.groups()
            return party1 in result_name and party2 in result_name
        return query in result_name


def enhanced_scan_with_rag(text: str, extracted_citations: List[Dict]) -> Dict:
    """
    Verify extracted citations against CourtListener database.
    Returns verification results with detailed context.
    """
    rag = LegalRAG()
    
    verified = []
    unverified = []
    context_parts = []
    
    citation_limit = min(len(extracted_citations), 15)
    
    for cite in extracted_citations[:citation_limit]:
        cite_type = cite.get("type", "")
        content = cite.get("content", "")
        
        if not content or len(content) < 5:
            continue
        
        logger.info(f"Verifying: {content} (type: {cite_type})")
        
        if cite_type in ("case", "case_citation"):
            result = rag.verify_citation(content)
        elif cite_type == "case_name":
            result = rag.verify_case_name(content)
        else:
            result = rag.verify_citation(content)
        
        if result.get("verified"):
            verified.append({
                "content": content,
                "type": cite_type,
                "case_name": result.get("case_name"),
                "court": result.get("court"),
                "date_filed": result.get("date_filed"),
                "confidence": result.get("confidence"),
                "search_method": result.get("search_method")
            })
            context_parts.append(
                f"VERIFIED: {content} is {result.get('case_name')} "
                f"({result.get('court')}, {result.get('date_filed')})"
            )
        else:
            unverified.append({
                "content": content,
                "type": cite_type,
                "reason": "Not found in CourtListener database"
            })
            context_parts.append(f"NOT FOUND: {content} could not be verified")
    
    return {
        "verified": verified,
        "unverified": unverified,
        "rag_context": "\n".join(context_parts),
        "total_checked": len(verified) + len(unverified),
        "verification_rate": len(verified) / max(len(verified) + len(unverified), 1) * 100
    }


if __name__ == "__main__":
    test_citations = [
        {"type": "case", "content": "347 U.S. 483"},
        {"type": "case", "content": "410 U.S. 113"},
        {"type": "case", "content": "384 U.S. 436"},
        {"type": "case", "content": "999 F.2d 999"},
        {"type": "case_name", "content": "Brown v Board of Education"},
    ]
    
    print("CourtListener Citation Verification Test")
    print("=" * 60)
    
    result = enhanced_scan_with_rag("", test_citations)
    
    print(f"\nVerified Citations: {len(result['verified'])}")
    for v in result['verified']:
        print(f"  [OK] {v['content']}")
        print(f"       Case: {v['case_name']}")
        print(f"       Court: {v['court']}, Date: {v['date_filed']}")
    
    print(f"\nUnverified Citations: {len(result['unverified'])}")
    for u in result['unverified']:
        print(f"  [--] {u['content']}")
        print(f"       Reason: {u['reason']}")
    
    print(f"\nVerification Rate: {result['verification_rate']:.1f}%")