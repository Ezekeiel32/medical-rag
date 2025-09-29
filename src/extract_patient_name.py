"""
Extract patient name from OCR results using pattern matching and NLP.
"""

import re
import json
from typing import Optional, Dict, Any, List


def extract_patient_name(ocr_results: Dict[str, Any]) -> Optional[str]:
    """
    Extract patient name from OCR results.
    Looks for patterns like "שם המבוטח:", "שם:", "Patient Name:", etc.
    
    Args:
        ocr_results: OCR results dict with pages and text
    
    Returns:
        Extracted patient name or None
    """
    # Common patterns for patient name in Hebrew medical documents
    name_patterns = [
        # Hebrew patterns
        r'שם\s*(?:ה)?מבוטח[:\s]+([^\n]+)',  # שם המבוטח: or שם מבוטח:
        r'שם\s*(?:ה)?חולה[:\s]+([^\n]+)',    # שם החולה: or שם חולה:
        r'שם\s*(?:ה)?מטופל[:\s]+([^\n]+)',   # שם המטופל: or שם מטופל:
        r'שם\s*פרטי\s*ומשפחה[:\s]+([^\n]+)', # שם פרטי ומשפחה:
        r'שם[:\s]+([^\n]+)',                  # שם:
        r'מבוטח[:\s]+([^\n]+)',               # מבוטח:
        # English patterns (in case of mixed documents)
        r'Patient\s*Name[:\s]+([^\n]+)',
        r'Name[:\s]+([^\n]+)',
        # Pattern for name after ID number
        r'ת\.?ז\.?[:\s]*\d{9}\s+([^\n]+)',   # ת.ז: 123456789 שם המבוטח
    ]
    
    # Try to find name in the first few pages
    pages = ocr_results.get('pages', [])
    for page_idx in range(min(3, len(pages))):  # Check first 3 pages
        page = pages[page_idx]
        text = page.get('text', '')
        
        # Also check line-by-line for better accuracy
        lines = page.get('lines', [])
        
        # First try on full page text
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.UNICODE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = clean_patient_name(name)
                if name and is_valid_name(name):
                    return name
        
        # Then try line by line
        for line in lines:
            line_text = line.get('text', '')
            for pattern in name_patterns:
                match = re.search(pattern, line_text, re.UNICODE)
                if match:
                    name = match.group(1).strip()
                    name = clean_patient_name(name)
                    if name and is_valid_name(name):
                        return name
    
    # Fallback: Try to find any Hebrew name-like text in the first page
    if pages:
        first_page_text = pages[0].get('text', '')
        # Look for sequences of Hebrew letters that could be names
        hebrew_name_pattern = r'[\u05D0-\u05EA]+(?:\s+[\u05D0-\u05EA]+){1,3}'
        matches = re.findall(hebrew_name_pattern, first_page_text)
        
        for potential_name in matches:
            if is_valid_name(potential_name):
                return potential_name
    
    return None


def clean_patient_name(name: str) -> str:
    """
    Clean and normalize patient name.
    """
    # Remove common suffixes/prefixes
    name = re.sub(r'^\s*[-–—:]\s*', '', name)  # Remove leading dashes/colons
    name = re.sub(r'\s*[-–—:]\s*$', '', name)  # Remove trailing dashes/colons
    
    # Remove ID numbers that might be attached
    name = re.sub(r'\b\d{9}\b', '', name)  # Remove 9-digit ID
    name = re.sub(r'\bת\.?ז\.?\s*', '', name)  # Remove ת.ז
    
    # Remove common titles
    titles = ['מר', 'גב\'', 'גברת', 'ד"ר', 'דר\'', 'פרופ\'', 'Dr.', 'Mr.', 'Mrs.', 'Ms.']
    for title in titles:
        name = re.sub(r'\b' + re.escape(title) + r'\b\.?\s*', '', name, flags=re.IGNORECASE)
    
    # Remove punctuation except apostrophes (for names like ג'ון)
    name = re.sub(r'[^\u05D0-\u05EA\u0041-\u005A\u0061-\u007A\s\']', ' ', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def is_valid_name(name: str) -> bool:
    """
    Check if extracted text is likely a valid patient name.
    """
    if not name or len(name) < 2:
        return False
    
    # Should not be all numbers
    if name.replace(' ', '').isdigit():
        return False
    
    # Should not be too long (likely grabbed too much text)
    if len(name) > 50:
        return False
    
    # Should contain at least some Hebrew or English letters
    has_hebrew = bool(re.search(r'[\u05D0-\u05EA]', name))
    has_english = bool(re.search(r'[A-Za-z]', name))
    
    if not (has_hebrew or has_english):
        return False
    
    # Should not be common non-name words
    non_names = [
        'תאריך', 'מספר', 'טופס', 'דוח', 'רפואי', 'בדיקה', 'מסמך', 'עמוד',
        'date', 'number', 'form', 'report', 'medical', 'exam', 'document', 'page'
    ]
    
    name_lower = name.lower()
    for non_name in non_names:
        if non_name in name_lower:
            return False
    
    return True


def extract_patient_info(ocr_results: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Extract patient information including name and ID from OCR results.
    
    Returns:
        Dict with 'name' and 'id' keys
    """
    info = {
        'name': extract_patient_name(ocr_results),
        'id': extract_patient_id(ocr_results)
    }
    
    return info


def extract_patient_id(ocr_results: Dict[str, Any]) -> Optional[str]:
    """
    Extract patient ID number from OCR results.
    """
    # Pattern for Israeli ID (9 digits)
    id_patterns = [
        r'ת\.?ז\.?[:\s]*(\d{9})',           # ת.ז: 123456789
        r'מספר\s*זהות[:\s]*(\d{9})',        # מספר זהות: 123456789
        r'ת\.?\s*ז\.?\s*(\d{9})',           # ת ז 123456789
        r'\b(\d{9})\b',                     # Any 9-digit number
    ]
    
    pages = ocr_results.get('pages', [])
    for page_idx in range(min(3, len(pages))):  # Check first 3 pages
        page = pages[page_idx]
        text = page.get('text', '')
        
        for pattern in id_patterns:
            match = re.search(pattern, text)
            if match:
                id_num = match.group(1)
                # Validate it's a reasonable ID
                if id_num[0] != '0' and not id_num.startswith('00'):  # IDs don't usually start with 0
                    return id_num
    
    return None


if __name__ == "__main__":
    # Test with sample OCR results
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        info = extract_patient_info(results)
        print(f"Extracted patient name: {info['name']}")
        print(f"Extracted patient ID: {info['id']}")