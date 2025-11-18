import re
def remove_unwanted_patterns(line: str) -> str:
    """
    Clean noisy content and unnecessary patterns from PDF lines.
    """
    # Remove page numbers like "Page 3"
    line = re.sub(r'\bpage\s\d+\b', '', line, flags=re.IGNORECASE)

    # Remove URLs
    line = re.sub(r'https?://\S+|www.\S+', '', line)

    # Remove placeholder symbols and tags
    line = re.sub(r'[●]|[END]|[CONTINUED]', '', line)

    # Remove captions like "Chart 1:", "Figure 2:" (but NOT tables)
    line = re.sub(r'^(figure|chart)\s+\d+[:.-]?', '', line, flags=re.IGNORECASE)

    # Remove section or chapter headers like "SECTION II"
    line = re.sub(r'^(section|chapter)\s+[ivxlc\d]+\s[-–—]', '', line, flags=re.IGNORECASE)

    # Remove copyright lines
    line = re.sub(r'©.?\d{4}', '', line)

    # Remove lines with only special characters

    # Remove only square and curly brackets: [], {}
    

    # Normalize extra spaces
    return re.sub(r'\s+', ' ', line).strip()