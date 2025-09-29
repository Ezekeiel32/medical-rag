#!/usr/bin/env python3
"""
Answer Enhancement Module for Medical RAG System
Creates clear, coherent, and complete Hebrew medical responses using Gemini AI.
"""

import re
from typing import Dict, Any, List
from .llm_gemini import gemini_generate


def enhance_medical_answer(
    question: str,
    raw_answer: str,
    sources: List[Dict[str, Any]],
    language: str = "he"
) -> str:
    """
    Enhance a raw medical answer to be clear, coherent, and complete.
    
    Args:
        question: The original question
        raw_answer: Raw answer that may be fragmented
        sources: List of source documents
        language: Language code ("he" for Hebrew)
    
    Returns:
        Enhanced, coherent answer
    """
    
    if not raw_answer or not raw_answer.strip():
        return "לא נמצא מידע רלוונטי במקורות הזמינים."
    
    # Check if answer needs enhancement
    if not _needs_enhancement(raw_answer):
        return raw_answer.strip()
    
    # Create comprehensive enhancement prompt
    enhancement_prompt = _create_enhancement_prompt(question, raw_answer, sources)
    
    try:
        enhanced = gemini_generate(
            enhancement_prompt,
            system=_get_enhancement_system_prompt(),
            model="gemini-1.5-flash",
            temperature=0.3,
            json_only=True
        )
        
        if isinstance(enhanced, dict) and "answer" in enhanced:
            enhanced_text = enhanced["answer"].strip()
            
            # Validate the enhanced answer
            if _is_valid_answer(enhanced_text, question):
                return enhanced_text
        
    except Exception as e:
        print(f"Answer enhancement failed: {e}")
    
    # Fallback: try basic cleanup
    return _basic_cleanup(raw_answer)


def _needs_enhancement(answer: str) -> bool:
    """Check if an answer needs enhancement."""
    if not answer or len(answer.strip()) < 10:
        return True
    
    # Check for repetitive patterns (like "ממצאים מתאריך")
    if re.search(r'(ממצאים.*?מתאריך.*?){2,}', answer):
        return True
    
    # Check for date spam
    dates_count = len(re.findall(r'\d{1,2}/\d{1,2}/\d{4}', answer))
    if dates_count > 3:  # Too many dates without context
        return True
    
    # Check for incomplete sentences
    sentences = re.split(r'[.!?]+', answer.strip())
    complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(complete_sentences) < 2:
        return True
    
    # Check for fragmented text (too many short segments)
    words = answer.split()
    if len(words) < 15:  # Too short
        return True
    
    # Check for admin noise
    if re.search(r'\b\d{7,}\b|\d{2,3}-\d{6,7}', answer):
        return True
    
    # Check for repetitive clinical terms without context
    if answer.count("ממצאים") > 2 and len(answer) < 200:
        return True
    
    # Check for incomplete Hebrew text
    hebrew_chars = sum(1 for c in answer if '\u0590' <= c <= '\u05FF')
    if hebrew_chars < len(answer) * 0.6:  # Less than 60% Hebrew
        return True
    
    return False


def _is_valid_answer(answer: str, question: str) -> bool:
    """Validate that an enhanced answer is proper."""
    if not answer or len(answer.strip()) < 20:
        return False
    
    # Check for complete sentences
    if not re.search(r'[.!?]\s*$', answer.strip()):
        return False
    
    # Check Hebrew content
    hebrew_chars = sum(1 for c in answer if '\u0590' <= c <= '\u05FF')
    if hebrew_chars < len(answer) * 0.7:
        return False
    
    # Check for admin noise
    if re.search(r'\b\d{7,}\b|\d{2,3}-\d{6,7}', answer):
        return False
    
    return True


def _create_enhancement_prompt(question: str, raw_answer: str, sources: List[Dict[str, Any]]) -> str:
    """Create a comprehensive enhancement prompt."""
    
    # Summarize sources for context
    source_summaries = []
    for i, source in enumerate(sources[:3]):
        summary = {
            "מקור": i + 1,
            "סוג_מסמך": source.get("document_type", "לא צוין"),
            "תאריך": source.get("document_date", "לא צוין"),
            "ציטוט_מרכזי": source.get("quote", "")[:300]
        }
        source_summaries.append(summary)
    
    return f"""
נושא השיפור: יצירת תשובה רפואית מקיפה וברורה

השאלה המקורית:
{question}

התשובה הגולמית שצריכה שיפור:
{raw_answer}

מקורות רפואיים זמינים:
{chr(10).join(f"מקור {i+1}: {s.get('סוג_מסמך', '')} מתאריך {s.get('תאריך', '')}" for i, s in enumerate(source_summaries))}

משימתך:
1. צור תשובה מקיפה וקוהרנטית בעברית שעונה על השאלה במלואה
2. התשובה חייבת להיות כתובה במשפטים מלאים וברורים
3. כלול את כל המידע הרלוונטי מהמקורות בצורה מאורגנת
4. אסור לכלול פרטים מנהליים, מספרי טלפון או מספרי זהות
5. התשובה תהיה בת 4-10 משפטים מלאים
6. כל משפט יהיה מובן בפני עצמו ויתרום לתמונה הכוללת

דגשים חשובים:
- אין להמציא מידע שלא מופיע במקורות
- יש לוודא שכל משפט גמור ומובן
- התשובה תהיה קריאה ומקצועית
- יש להתמקד במידע הרלוונטי לשאלה

החזר JSON בפורמט:
{{"answer": "התשובה המקיפה והקוהרנטית בעברית"}}
"""


def _get_enhancement_system_prompt() -> str:
    """Get the system prompt for answer enhancement."""
    return """
אתה מומחה רפואי מנוסה המתמחה בעיבוד ושיפור תשובות רפואיות בעברית.

התמחותך היא:
• יצירת תשובות ברורות, מקיפות וקוהרנטיות
• כתיבה במשפטים מלאים ומובנים
• ארגון מידע רפואי באופן לוגי וקריא
• שמירה על דיוק עובדתי מוחלט

כללי עבודה:
• כתוב רק במשפטים מלאים וברורים
• ודא שכל משפט תורם לתמונה הכוללת
• אסור לכלול פרטים מנהליים או מספרים לא רלוונטיים
• התמקד במידע הרפואי החשוב והרלוונטי
• השתמש בעברית ברורה ומקצועית
"""


def _basic_cleanup(answer: str) -> str:
    """Basic cleanup for answers that couldn't be enhanced."""
    if not answer:
        return "לא נמצא מידע רלוונטי במקורות הזמינים."
    
    # Remove admin noise
    cleaned = re.sub(r'\b\d{7,}\b', '', answer)
    cleaned = re.sub(r'\d{2,3}-\d{6,7}', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Ensure proper ending
    cleaned = cleaned.strip()
    if cleaned and not re.search(r'[.!?]$', cleaned):
        cleaned += '.'
    
    return cleaned if cleaned else "לא נמצא מידע רלוונטי במקורות הזמינים."


__all__ = [
    "enhance_medical_answer"
]
