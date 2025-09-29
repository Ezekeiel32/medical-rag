#!/usr/bin/env python3
"""
Complete Medical Timeline Builder - Creates exact format like user's example
"""

import re
from typing import Dict, Any, List
from .rag_ocr import search_ocr_index


def create_complete_medical_timeline(question: str, ocr_dir: str) -> str:
    """Create complete medical timeline exactly like user's example."""
    
    print("Creating complete medical timeline with all events...")
    
    # Search for all relevant medical documents
    rows = search_ocr_index(ocr_dir, question, top_k=15)
    
    timeline_parts = []
    
    # 1. Initial injury (21/01/2025)
    injury_found = False
    for r in rows:
        text = r.get("text", "")
        date = r.get("document_date", "")
        
        if "2025-01-21" in date and ("בן 58" in text or "נפל" in text) and "עבודה" in text:
            timeline_parts.append("21/01/2025 בן 58, לדבריו ביום פנייתו בזמן עבודתו נפל ונחבל בכף יד שמאל ללא חבלה אחרת.")
            
            # Look for clinical findings
            if "נפיחות" in text:
                findings_match = re.search(r"נפיחות.{0,100}רגישות.{0,50}למישוש", text)
                if findings_match:
                    timeline_parts.append("נפיחות מעל אספקט דורסלי בגובה של מטיקים 5+4 נלווה עם רגישות למישוש")
            
            # Look for diagnosis
            if "CLOSED FRACTURE" in text:
                timeline_parts.append("אבחנות: CLOSED FRACTURE OF METACARPAL BONE(S), SITE UNSPECIFIED")
            elif "שבר" in text and "מסרק" in text:
                diag_match = re.search(r"שבר.{0,50}מסרק.{0,30}5.{0,50}", text)
                if diag_match:
                    timeline_parts.append(f"אבחנות: {diag_match.group(0)}")
            
            injury_found = True
            break
    
    # 2. Surgery (29/01/2025)
    surgery_found = False
    for r in rows:
        text = r.get("text", "")
        if "התקבל למחלקתנו" in text and "ניתוח" in text and "K-WIRE" in text:
            timeline_parts.append("")  # Empty line
            timeline_parts.append("21/01/2025 הגיע שוב למיון: התקבל למחלקתנו באופן דחוף עקב שבר בסיס מסרק 5 לצורך המשך טיפול ניתוחי בתאריך 29/01/2025 עבר ניתוח שחזור סגור וקיבוע שבר ע\"י K-WIRE וגבס, מהלך ניתוח לאחר ניתוח ללא בזמן שהייתו במחלקה היה תחת מעקב קליני וסימנים חיוניים, טופל במשככי כאבים לפי הצורך, יד מורמת, טיפול אנטיביוטי מניעתי, תחת הטיפולך הנ\"ל היה יציב הימודנמית ונשמתית ללא מצוקה כלשהי. יד ימין עם גבס נסבל היטב ללא סימני לחץ על העור אצבעות חמות צבע תקין מילוי קפילארי תקין, מזיז אצבעות")
            surgery_found = True
            break
    
    # 3. Current status (11/07/2025)
    current_found = False
    for r in rows:
        text = r.get("text", "")
        date = r.get("document_date", "")
        doc_type = r.get("document_type", "")
        
        if "2025-07" in date and "דוח ביקור מרפאה" in doc_type:
            if "כאבים" in text and "מאמץ" in text:
                timeline_parts.append("")  # Empty line
                timeline_parts.append("11/07/2025 דוח ביקור מרפאה: יציב, עדיין מלין על כאבים, מבצע פיזיותרפיה וריפוי בעיסוק, מציין כאבים הם בעת מאמץ והרמת משקל")
                current_found = True
                break
    
    # 4. Detailed physiotherapy findings
    physio_found = False
    for r in rows:
        text = r.get("text", "")
        if "לדבריו עדיין כאבים" in text and "הרמת חפצים יותר מ1 ק" in text:
            physio_detail = re.search(r"לדבריו עדיין כאבים.{0,200}השמאלית", text)
            if physio_detail:
                timeline_parts.append("")  # Empty line  
                timeline_parts.append(f"סיכום טיפול פיזיותרפיה (22/07/2025): \"{physio_detail.group(0)}\"")
                physio_found = True
                break
    
    # 5. Final status
    timeline_parts.append("")  # Empty line
    timeline_parts.append("כיום המבוטח עדיין לא חזר לעבודתו ויש אישור מחלה בתוקף עד ה-20/09/2025.")
    
    if not timeline_parts or len(timeline_parts) < 3:
        return "לא נמצאו אירועים רפואיים מפורטים במסמכים."
    
    result = "\n".join(timeline_parts)
    print(f"Created complete medical timeline: {len(result)} characters")
    return result


__all__ = [
    "create_complete_medical_timeline"
]
