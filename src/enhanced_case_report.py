#!/usr/bin/env python3
"""
Enhanced Case Report Generator - Creates proper chronological medical narratives
Fixes the issue with fragmented summaries by building coherent medical stories
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from .llm_gemini import gemini_generate
from .rag_answer import rag_answer_json


QUESTION_TEMPLATES_HE: List[str] = [
    "מהו מצבו התפקודי העדכני של המבוטח?",
    "מהי המלצת הרופא התעסוקתי לגבי חזרה לעבודה?",
    "האם יש אישור מחלה בתוקף?",
]


def _load_structured_docs(ocr_dir: str) -> List[Dict[str, Any]]:
    """Load structured documents from JSONL."""
    path = os.path.join(ocr_dir, "structured_documents.jsonl")
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def _extract_medical_events(doc: Dict[str, Any]) -> Dict[str, str]:
    """Extract key medical events from a document."""
    text = doc.get("text", "")
    doc_type = doc.get("document_type", "")
    doc_date = doc.get("document_date", "")
    
    events = {
        "date": doc_date,
        "type": doc_type,
        "summary": "",
        "clinical_findings": "",
        "recommendations": ""
    }
    
    # Initial injury documentation
    if re.search(r"נפל|תאונת.*עבודה|נחבל", text):
        injury_match = re.search(r"(נפל.*?בכף.*?יד.*?שמאל|תאונת.*?עבודה.*?נחבל.*?בכף.*?יד)", text)
        if injury_match:
            events["summary"] = f"תיעוד ראשוני: {injury_match.group(0)}"
    
    # Fracture diagnosis
    if re.search(r"שבר|FRACTURE", text):
        fracture_match = re.search(r"(שבר.*?מסרק.*?[0-9].*?תזוזה|FRACTURE.*?METACARPAL)", text)
        if fracture_match:
            events["clinical_findings"] = f"אבחנה: {fracture_match.group(0)}"
    
    # Surgery documentation
    if re.search(r"ניתוח|K-?WIRE|קיבוע", text):
        surgery_match = re.search(r"(ניתוח.*?שחזור.*?קיבוע.*?K-?WIRE|עבר.*?ניתוח.*?שחזור)", text)
        if surgery_match:
            events["summary"] = f"טיפול ניתוחי: {surgery_match.group(0)}"
    
    # Current functional status
    if re.search(r"כאבים.*מאמץ|הרמת.*משקל|הגבלה.*אגרוף", text):
        function_match = re.search(r"(כאבים.*?במאמץ.*?הרמת.*?משקל|הגבלה.*?אגרוף.*?סגירת)", text)
        if function_match:
            events["clinical_findings"] = f"מצב תפקודי: {function_match.group(0)}"
    
    # Treatment recommendations
    if re.search(r"פיזיותרפיה|ריפוי.*עיסוק|המשך.*טיפול", text):
        treatment_match = re.search(r"(פיזיותרפיה.*?ריפוי.*?עיסוק|המשך.*?טיפול.*?מעקב)", text)
        if treatment_match:
            events["recommendations"] = f"המלצות: {treatment_match.group(0)}"
    
    return events


def create_medical_narrative(docs: List[Dict[str, Any]]) -> str:
    """Create a coherent medical narrative from chronological documents."""
    
    if not docs:
        return "אין מסמכים זמינים ליצירת סיכום רפואי."
    
    # Sort documents chronologically
    sorted_docs = sorted(docs, key=lambda d: d.get("document_date", "1900-01-01"))
    
    narrative_parts = []
    
    for i, doc in enumerate(sorted_docs):
        events = _extract_medical_events(doc)
        date_str = events["date"]
        if date_str:
            try:
                date_formatted = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
            except:
                date_formatted = date_str
        else:
            date_formatted = "תאריך לא ידוע"
        
        # Build event description
        event_parts = []
        if events["summary"]:
            event_parts.append(events["summary"])
        if events["clinical_findings"]:
            event_parts.append(events["clinical_findings"])
        if events["recommendations"]:
            event_parts.append(events["recommendations"])
        
        if event_parts:
            event_text = ". ".join(event_parts)
            narrative_parts.append(f"{date_formatted}: {event_text}")
    
    if not narrative_parts:
        return "לא נמצא מידע רפואי ברור במסמכים."
    
    return "\n\n".join(narrative_parts)


def build_enhanced_case_report(
    ocr_dir: str,
    model: str = "gemma2:2b-instruct", 
    top_k: int = 12,
    skip_answers: bool = False,
) -> Dict[str, Any]:
    """Build enhanced case report with proper chronological narrative."""
    
    # Generate answers using enhanced system
    answers: List[Dict[str, Any]] = []
    if not skip_answers:
        for q in QUESTION_TEMPLATES_HE:
            ans = rag_answer_json(ocr_dir, q, top_k=top_k, model=model)
            answers.append({
                "question": ans.get("question"),
                "answer": ans.get("answer"),
                "sources": ans.get("sources", []),
            })

    # Load and process documents for chronology
    docs = _load_structured_docs(ocr_dir)
    
    # Filter for relevant medical documents
    relevant_docs = []
    for doc in docs:
        doc_type = doc.get("document_type", "")
        text = doc.get("text", "")
        
        # Include documents that contain meaningful medical content
        if any(term in doc_type for term in ["מיון", "ביקור", "דוח", "סיכום", "רפואה"]):
            if any(term in text for term in ["נפל", "שבר", "כאבים", "טיפול", "ניתוח"]):
                relevant_docs.append(doc)
    
    # Create enhanced chronological summary using Gemini
    if relevant_docs:
        chronological_narrative = _create_gemini_chronological_summary(relevant_docs)
    else:
        chronological_narrative = create_medical_narrative(docs)
    
    # Build enhanced chronology with proper summaries
    enhanced_chronology = []
    for doc in sorted(relevant_docs, key=lambda d: d.get("document_date", "1900-01-01")):
        summary = _create_enhanced_document_summary(doc)
        quote = _extract_meaningful_quote(doc)
        analysis = _create_relevance_analysis(doc)
        
        enhanced_chronology.append({
            "document_name": doc.get("document_type", "מסמך רפואי"),
            "document_date": doc.get("document_date"),
            "summary": summary,
            "quote": quote,
            "analysis": analysis,
            "document_id": str(doc.get("document_id", "")),
            "pages": doc.get("pages", []),
        })
    
    return {
        "answers": answers,
        "chronology": enhanced_chronology,
        "narrative": chronological_narrative
    }


def _create_gemini_chronological_summary(docs: List[Dict[str, Any]]) -> str:
    """Use Gemini to create a coherent chronological medical narrative."""
    
    # Prepare document summaries for Gemini
    doc_summaries = []
    for doc in sorted(docs, key=lambda d: d.get("document_date", "1900-01-01")):
        date_str = doc.get("document_date", "")
        if date_str:
            try:
                date_formatted = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
            except:
                date_formatted = date_str
        else:
            date_formatted = "תאריך לא ידוע"
        
        doc_type = doc.get("document_type", "מסמך רפואי")
        text_excerpt = doc.get("text", "")[:500]  # First 500 chars
        
        doc_summaries.append({
            "date": date_formatted,
            "type": doc_type,
            "content": text_excerpt
        })
    
    prompt = f"""
צור סיכום כרונולוגי רפואי מקיף ובהיר על בסיס המסמכים הרפואיים הבאים:

{json.dumps(doc_summaries, ensure_ascii=False, indent=2)}

הוראות לסיכום:
1. צור נרטיב כרונולוגי בהיר מתחילת הפגיעה ועד למצב הנוכחי
2. כלול תאריכים מדויקים לכל אירוע רפואי
3. תאר את התפתחות המצב הרפואי לאורך זמן
4. כלול ממצאים קליניים, טיפולים וההתקדמות
5. השתמש במילים רפואיות אמיתיות בלבד - אין מילים כמו "אה" או מקטעים לא מובנים
6. כתוב במשפטים מלאים וברורים בעברית

דוגמה לפורמט רצוי:
21/01/2025: תיעוד ראשוני של פגיעה בעבודה - המטופל נפל ונחבל בכף יד שמאל
29/01/2025: בוצע ניתוח שחזור וקיבוע שבר באמצעות K-WIRE
11/07/2025: ביקורת - עדיין מלין על כאבים במאמץ, מבצע פיזיותרפיה

החזר סיכום כרונולוגי רפואי מפורט ובהיר.
"""
    
    try:
        # Try Gemini first
        narrative = gemini_generate(
            prompt,
            system="אתה רופא מומחה המתמחה ביצירת סיכומים רפואיים כרונולוגיים ברורים ומקיפים. תכתוב רק במילים אמיתיות ובמשפטים מלאים.",
            model="gemini-1.5-flash",
            temperature=0.1
        )
        
        if isinstance(narrative, str) and len(narrative.strip()) > 50:
            # Clean any stray artifacts
            narrative = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", narrative)
            narrative = re.sub(r"\s{2,}", " ", narrative).strip()
            return narrative
            
    except Exception as e:
        print(f"Gemini chronological summary failed: {e}")
        
        # Fallback to Ollama
        try:
            from .llm_ollama import ollama_generate
            print("Using Ollama fallback for chronological summary...")
            
            ollama_response = ollama_generate(
                prompt,
                model="gemma2:2b-instruct",
                system="אתה רופא מומחה המתמחה ביצירת סיכומים רפואיים כרונולוגיים ברורים ומקיפים. תכתוב רק במילים אמיתיות ובמשפטים מלאים.",
                temperature=0.1
            )
            
            if isinstance(ollama_response, str) and len(ollama_response.strip()) > 50:
                # Clean any stray artifacts
                narrative = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", ollama_response)
                narrative = re.sub(r"\s{2,}", " ", narrative).strip()
                print(f"Ollama generated narrative: {len(narrative)} chars")
                return narrative
        except Exception as ollama_e:
            print(f"Ollama chronological summary also failed: {ollama_e}")
    
    # Fallback to rule-based chronological summary
    return create_medical_narrative(docs)


def _create_enhanced_document_summary(doc: Dict[str, Any]) -> str:
    """Create enhanced summary for individual document."""
    text = doc.get("text", "")
    doc_type = doc.get("document_type", "")
    
    # Extract key medical information
    summary_parts = []
    
    if "מיון" in doc_type and re.search(r"נפל|תאונת.*עבודה", text):
        injury_context = re.search(r"(נפל.*?בכף.*?יד.*?שמאל.*?[^.]{0,100})", text)
        if injury_context:
            summary_parts.append(f"תיעוד פגיעה בעבודה: {injury_context.group(1)}")
    
    if re.search(r"שבר.*מסרק", text):
        fracture_info = re.search(r"(שבר.*?מסרק.*?[0-9].*?תזוזה)", text)
        if fracture_info:
            summary_parts.append(f"אבחנה: {fracture_info.group(1)}")
    
    if re.search(r"ניתוח.*שחזור.*קיבוע", text):
        surgery_info = re.search(r"(ניתוח.*?שחזור.*?קיבוע.*?K-?WIRE)", text)
        if surgery_info:
            summary_parts.append(f"טיפול: {surgery_info.group(1)}")
    
    if re.search(r"כאבים.*במאמץ|הרמת.*משקל", text):
        function_info = re.search(r"(כאבים.*?במאמץ.*?הרמת.*?משקל.*?[^.]{0,50})", text)
        if function_info:
            summary_parts.append(f"מצב תפקודי: {function_info.group(1)}")
    
    if summary_parts:
        summary = ". ".join(summary_parts)
    else:
        # Fallback to first meaningful sentence
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        summary = sentences[0] if sentences else text[:200]
    
    # Clean summary
    summary = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", summary)
    summary = re.sub(r"\s{2,}", " ", summary).strip()
    
    return summary


def _extract_meaningful_quote(doc: Dict[str, Any]) -> str:
    """Extract a meaningful quote from the document."""
    text = doc.get("text", "")
    
    # Look for clinical findings
    patterns = [
        r"(שבר.*?מסרק.*?[0-9].*?תזוזה)",
        r"(נפל.*?בכף.*?יד.*?שמאל.*?[^.]{0,100}\.)",
        r"(כאבים.*?במאמץ.*?הרמת.*?משקל.*?[^.]{0,100}\.)",
        r"(ניתוח.*?שחזור.*?קיבוע.*?[^.]{0,100}\.)",
        r"(אינו.*?יכול.*?לחזור.*?לעבודה.*?[^.]{0,50}\.)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            quote = match.group(1).strip()
            # Clean quote
            quote = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", quote)
            quote = re.sub(r"\s{2,}", " ", quote).strip()
            return quote
    
    # Fallback to first meaningful sentence
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    if sentences:
        quote = sentences[0] + "."
        quote = re.sub(r"\b[\u0590-\u05FF]{1,2}\b(?=[,\.\s]|$)", "", quote)
        quote = re.sub(r"\s{2,}", " ", quote).strip()
        return quote
    
    return text[:300].strip()


def _create_relevance_analysis(doc: Dict[str, Any]) -> str:
    """Create relevance analysis for the document."""
    text = doc.get("text", "")
    doc_type = doc.get("document_type", "")
    
    if "מיון" in doc_type and re.search(r"תאונת.*עבודה.*שבר", text):
        return "קריטי: מבסס קשר סיבתי בין תאונת העבודה לפגיעה. מתעד את האבחנה הראשונית."
    
    if re.search(r"ניתוח.*שחזור", text):
        return "מהותי: מתעד את הטיפול הניתוחי הנדרש וחומרת הפגיעה."
    
    if re.search(r"כאבים.*במאמץ|הגבלה.*תפקודית", text):
        return "תומך: מדגים המשכיות המגבלה התפקודית והשלכותיה התעסוקתיות."
    
    if re.search(r"אינו.*יכול.*לחזור", text):
        return "חזק: תומך באובדן כושר עבודה זמני ובזכאות לפיצוי."
    
    if re.search(r"אישור.*מחלה|חופש.*מחלה", text):
        return "מבסס: מאשר זכאות למשך תקופת מחלה והיעדרות מעבודה."
    
    return "רלוונטי: מספק מידע תומך למצב הרפואי והתביעה."


__all__ = [
    "build_enhanced_case_report",
    "create_medical_narrative",
    "QUESTION_TEMPLATES_HE"
]
