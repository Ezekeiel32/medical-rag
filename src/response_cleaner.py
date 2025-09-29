#!/usr/bin/env python3
"""
Response Cleaner - Aggressively removes JSON artifacts and ensures clean Hebrew medical responses
"""

import re
import json
from typing import Any, Dict


def clean_medical_response(response_text: str) -> str:
    """Aggressively clean medical response from JSON artifacts and formatting."""
    
    if not response_text or not response_text.strip():
        return "לא נמצא מידע רלוונטי במקורות הזמינים."
    
    clean_text = response_text.strip()
    
    # Remove JSON code block formatting
    clean_text = re.sub(r'```json\s*', '', clean_text)
    clean_text = re.sub(r'```\s*', '', clean_text)
    
    # Extract from JSON structure if present
    try:
        if clean_text.startswith('{') and clean_text.endswith('}'):
            parsed = json.loads(clean_text)
            if "answer" in parsed:
                clean_text = parsed["answer"].strip()
        elif '{"answer"' in clean_text:
            start = clean_text.find('{"answer"')
            end = clean_text.rfind('}') + 1
            if start != -1 and end > start:
                json_part = clean_text[start:end]
                parsed = json.loads(json_part)
                if "answer" in parsed:
                    clean_text = parsed["answer"].strip()
    except:
        pass
    
    # Remove remaining JSON artifacts
    clean_text = re.sub(r'^{.*?"answer":\s*"', '', clean_text)
    clean_text = re.sub(r'"\s*}\.?$', '', clean_text)
    clean_text = re.sub(r'^"', '', clean_text)  # Leading quote
    clean_text = re.sub(r'"$', '', clean_text)  # Trailing quote
    
    # Remove stray Hebrew artifacts
    clean_text = re.sub(r'\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)', '', clean_text)
    
    # Clean up spacing and punctuation
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'\s+([.,!?])', r'\1', clean_text)
    
    # Ensure proper ending
    clean_text = clean_text.strip()
    if clean_text and not clean_text.endswith(('.', '!', '?')):
        clean_text += '.'
    
    # Final validation
    if len(clean_text) < 20:
        return "לא נמצא מידע רלוונטי במקורות הזמינים."
    
    return clean_text


def clean_case_report_answers(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all answers in a case report."""
    
    if "answers" in report_data:
        for answer_item in report_data["answers"]:
            if "answer" in answer_item:
                original = answer_item["answer"]
                cleaned = clean_medical_response(original)
                answer_item["answer"] = cleaned
                
    return report_data


__all__ = [
    "clean_medical_response",
    "clean_case_report_answers"
]
