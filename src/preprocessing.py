"""Content filtering for RAG quality."""
import re

def should_skip_chunk(text: str) -> bool:
    """
    Filter out non-semantic chunks (bibliography, headers, etc).
    
    Returns True if chunk should be skipped.
    """
    # Skip if mostly references/URLs
    if text.count("http://") + text.count("https://") > 2:
        return True
    
    # Skip if mostly citations (Author (YEAR) pattern)
    citation_pattern = r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\s+\(\d{4}\)'
    if len(re.findall(citation_pattern, text)) > 3:
        return True
    
    # Skip if very short
    if len(text.strip()) < 100:
        return True
    
    # Skip common bibliography headers
    bib_keywords = ["literaturverzeichnis", "references", "bibliography", 
                    "quellenverzeichnis", "letzter zugriff"]
    if any(kw in text.lower() for kw in bib_keywords):
        return True
    
    return False