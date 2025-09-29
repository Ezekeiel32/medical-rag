"""
Advanced Query Intelligence and Response Enhancement System
Provides intelligent query understanding, prompt improvement, and response decoding
for comprehensive medical RAG responses.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from .llm_ollama import ollama_generate
from .llm_gemini import gemini_generate
from .timezone_utils import get_jerusalem_date_string


class QueryIntelligenceSystem:
    """Intelligent query understanding and enhancement system."""
    
    def __init__(self):
        # Query type patterns
        self.temporal_patterns = {
            'monthly_summary': re.compile(r'(?:סכם|תסכם|סיכום).*(?:חודש|בחודש)\s*(\d{1,2})', re.IGNORECASE),
            'date_range': re.compile(r'(?:בין|מ).*?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}).*?(?:ל|עד).*?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})', re.IGNORECASE),
            'specific_date': re.compile(r'(?:ב|בתאריך)\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})', re.IGNORECASE),
            'recent_events': re.compile(r'(?:לאחרונה|בתקופה האחרונה|בזמן האחרון|מה קרה לאחרונה)', re.IGNORECASE),
            'progression': re.compile(r'(?:התפתחות|התקדמות|שינוי|מה השתנה)', re.IGNORECASE)
        }
        
        self.biographical_patterns = {
            'birth_date': re.compile(r'(?:מתי.*נולד|תאריך.*לידה|גיל|בן.*כמה)', re.IGNORECASE),
            'personal_info': re.compile(r'(?:מידע.*אישי|פרטים.*אישיים|זהות)', re.IGNORECASE),
            'contact_info': re.compile(r'(?:כתובת|טלפון|פרטי.*קשר)', re.IGNORECASE),
            'employment': re.compile(r'(?:עבודה|מקצוע|תפקיד|מעסיק)', re.IGNORECASE)
        }
        
        self.medical_patterns = {
            'diagnosis': re.compile(r'(?:אבחנה|מחלה|בעיה.*רפואית|מה.*יש.*לו)', re.IGNORECASE),
            'treatment': re.compile(r'(?:טיפול|תרופות|ניתוח|פעולה)', re.IGNORECASE),
            'symptoms': re.compile(r'(?:תסמינים|כאבים|תלונות|מה.*מרגיש)', re.IGNORECASE),
            'prognosis': re.compile(r'(?:תחזית|פרוגנוזה|מה.*צפוי)', re.IGNORECASE),
            'functional_status': re.compile(r'(?:מצב.*תפקודי|יכולת.*תפקוד|הגבלות)', re.IGNORECASE)
        }
        
        self.administrative_patterns = {
            'insurance': re.compile(r'(?:ביטוח|תביעה|פיצוי|זכאות)', re.IGNORECASE),
            'work_status': re.compile(r'(?:חזרה.*לעבודה|כושר.*עבודה|מנוחה.*מעבודה)', re.IGNORECASE),
            'certificates': re.compile(r'(?:תעודה|אישור|מחלה)', re.IGNORECASE)
        }

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and extract key information."""
        
        intent_analysis = {
            'query_type': 'general',
            'temporal_scope': None,
            'specific_dates': [],
            'month_filter': None,
            'biographical_request': None,
            'medical_focus': [],
            'administrative_focus': [],
            'requires_summary': False,
            'requires_chronology': False,
            'search_keywords': []
        }
        
        # Detect temporal queries
        for pattern_name, pattern in self.temporal_patterns.items():
            match = pattern.search(query)
            if match:
                intent_analysis['query_type'] = 'temporal'
                intent_analysis['temporal_scope'] = pattern_name
                
                if pattern_name == 'monthly_summary':
                    intent_analysis['month_filter'] = int(match.group(1))
                    intent_analysis['requires_summary'] = True
                    intent_analysis['requires_chronology'] = True
                elif pattern_name in ['date_range', 'specific_date']:
                    intent_analysis['specific_dates'] = [match.group(1)]
                    if pattern_name == 'date_range' and len(match.groups()) > 1:
                        intent_analysis['specific_dates'].append(match.group(2))
                break
        
        # Detect biographical queries
        for bio_type, pattern in self.biographical_patterns.items():
            if pattern.search(query):
                intent_analysis['query_type'] = 'biographical'
                intent_analysis['biographical_request'] = bio_type
                break
        
        # Detect medical focus areas
        for med_type, pattern in self.medical_patterns.items():
            if pattern.search(query):
                intent_analysis['medical_focus'].append(med_type)
        
        # Detect administrative focus areas
        for admin_type, pattern in self.administrative_patterns.items():
            if pattern.search(query):
                intent_analysis['administrative_focus'].append(admin_type)
        
        # Extract search keywords
        intent_analysis['search_keywords'] = self._extract_keywords(query)
        
        # Determine if summary/chronology needed
        if any(word in query for word in ['סכם', 'סיכום', 'תסכם', 'מה קרה']):
            intent_analysis['requires_summary'] = True
            
        if any(word in query for word in ['התפתחות', 'התקדמות', 'לאורך זמן', 'בזמן']):
            intent_analysis['requires_chronology'] = True
        
        return intent_analysis

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful Hebrew keywords from query."""
        
        # Common medical Hebrew keywords
        medical_keywords = [
            'כאבים', 'שבר', 'פגיעה', 'תאונה', 'ניתוח', 'טיפול', 'תרופות',
            'אבחנה', 'בדיקה', 'צילום', 'מיון', 'בית חולים', 'רופא', 'מרפאה',
            'פיזיותרפיה', 'ריפוי בעיסוק', 'שיקום', 'התאוששות', 'שיפור', 'הרעה',
            'תפקוד', 'יכולת', 'הגבלה', 'מגבלה', 'עבודה', 'מעסיק', 'ביטוח',
            'תביעה', 'פיצוי', 'נכות', 'אישור', 'תעודה', 'מחלה'
        ]
        
        keywords = []
        words = query.split()
        
        for word in words:
            clean_word = re.sub(r'[^\u0590-\u05FF\w]', '', word)  # Keep Hebrew and alphanumeric
            if len(clean_word) >= 2 and clean_word in medical_keywords:
                keywords.append(clean_word)
        
        return list(set(keywords))  # Remove duplicates

    def enhance_query(self, query: str, intent_analysis: Dict[str, Any]) -> str:
        """Enhance the original query with intelligent context and specificity."""
        
        enhanced_query = query.strip()
        
        if intent_analysis['query_type'] == 'temporal' and intent_analysis['month_filter']:
            # Add temporal context for monthly summaries
            month = intent_analysis['month_filter']
            enhanced_query = f"סכם את כל האירועים הרפואיים והטיפולים שקרו למטופל בחודש {month:02d}. כלול תאריכים מדויקים, סוג הטיפול, ממצאים קליניים, והמלצות רפואיות. הסתמך רק על מסמכים מהתקופה הרלוונטית."
            
        elif intent_analysis['query_type'] == 'biographical':
            bio_type = intent_analysis['biographical_request']
            if bio_type == 'birth_date':
                enhanced_query = "מצא את תאריך הלידה של המטופל או גילו. חפש במסמכי זהות, טפסי קבלה, או כל מקום שמופיע מידע אישי."
            elif bio_type == 'employment':
                enhanced_query = "מצא מידע על עבודתו של המטופל: מקום העבודה, תפקיד, מעסיק. חפש במסמכי ביטוח, דוחות תאונת עבודה ותעסוקה רפואית."
        
        elif intent_analysis['medical_focus']:
            # Enhance medical queries with specific medical context
            medical_context = []
            for focus in intent_analysis['medical_focus']:
                if focus == 'diagnosis':
                    medical_context.append("כלול אבחנות מדויקות, קודי ICD אם זמינים")
                elif focus == 'treatment':
                    medical_context.append("פרט סוגי טיפול, תרופות, מינון, תדירות")
                elif focus == 'symptoms':
                    medical_context.append("תאר תסמינים ברמת פירוט גבוהה כולל עוצמה ותדירות")
                elif focus == 'functional_status':
                    medical_context.append("כלול מדידות תפקוד, מגבלות ספציפיות, הערכות טווח תנועה")
            
            if medical_context:
                enhanced_query += f" {' '.join(medical_context)}. בסס את כל המידע על מסמכים רפואיים בלבד."
        
        # Add comprehensive search instruction
        enhanced_query += " חפש בכל המסמכים הזמינים ובנה תשובה מקיפה ומדויקת."
        
        return enhanced_query

    def create_smart_prompt(self, enhanced_query: str, context: str, intent_analysis: Dict[str, Any]) -> str:
        """Create an intelligent, context-aware prompt for the LLM."""
        
        today = get_jerusalem_date_string()
        
        # Base system instructions
        system_instructions = [
            "אתה מומחה רפואי שמנתח מסמכים רפואיים בעברית.",
            "ענה רק בעברית - אסור לענות בשפות אחרות כמו אנגלית או סינית.",
            "ענה רק על בסיס המידע במסמכים שסופקו.",
            "בנה תשובות מפורטות, קוהרנטיות ומלאות בעברית בלבד.",
            "השתמש בתאריכים בפורמט ישראלי (DD/MM/YYYY).",
            f"התאריך היום: {today}",
            "אם אין מידע זמין, ענה 'אין מידע זמין במסמכים על נושא זה' בעברית.",
        ]
        
        # Add specific instructions based on query type
        if intent_analysis['query_type'] == 'temporal':
            system_instructions.extend([
                "סדר אירועים בסדר כרונולוגי.",
                "צין תאריכים מדויקים לכל אירוע.",
                "הדגש שינויים במצב הרפואי לאורך זמן.",
            ])
            
        elif intent_analysis['query_type'] == 'biographical':
            system_instructions.extend([
                "חפש מידע אישי בקפדנות בכל המסמכים.",
                "אם המידע לא נמצא, ציין זאת בבירור.",
                "השתמש רק במידע מהמסמכים הרפואיים.",
            ])
        
        if intent_analysis['requires_summary']:
            system_instructions.extend([
                "צור סיכום מקיף של כל האירועים הרלוונטיים.",
                "ארגן את המידע באופן לוגי וקוהרנטי.",
                "הדגש נקודות מפתח וקשרים בין אירועים.",
            ])
        
        if intent_analysis['requires_chronology']:
            system_instructions.extend([
                "בנה ציר זמן ברור של האירועים.",
                "הראה התפתחות ושינויים לאורך זמן.",
                "קשר בין אירועים שונים אם רלוונטי.",
            ])
        
        # Construct the full prompt
        prompt = f"""
הנחיות מערכת:
{chr(10).join(f"• {instruction}" for instruction in system_instructions)}

שאלה: {enhanced_query}

הקשר ממסמכים רפואיים:
{context}

הוראות תשובה:
• ענה באופן מקיף ומפורט בעברית בלבד
• אסור לענות באנגלית, סינית או כל שפה אחרת
• בסס כל מידע על המסמכים בלבד
• ציין מקורות ותאריכים כאשר רלוונטי
• אל תמציא מידע שלא קיים במסמכים
• בנה משפטים מלאים וקוהרנטיים בעברית
• אם אין מידע זמין, ענה 'אין מידע זמין במסמכים על נושא זה'

תשובה בעברית:"""
        
        return prompt


class ResponseDecoder:
    """Advanced response decoder for coherent, complete medical responses."""
    
    def __init__(self):
        self.sentence_endings = re.compile(r'[.!?׃]')
        self.incomplete_patterns = [
            re.compile(r'\w+\s*$'),  # Text ending with partial word
            re.compile(r'(?:ו|או|כי|אשר|שיש|יש|לפי|על פי|נכון ל)\s*$'),  # Hebrew connectors at end
            re.compile(r'[,;:]\s*$'),  # Punctuation at end
        ]
        
        # Language validation patterns - more aggressive cleaning
        self.hebrew_pattern = re.compile(r'[\u0590-\u05FF]')  # Hebrew characters
        self.foreign_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]')  # Chinese, Japanese
        self.latin_pattern = re.compile(r'[a-zA-Z]{3,}')  # Long Latin sequences
        self.mixed_lang_pattern = re.compile(r'[^\u0590-\u05FF\s\d\.,;:!?\-()"\'\u200f\u200e]+')  # Any non-Hebrew, non-punctuation
    
    def clean_hebrew_response(self, response: str) -> str:
        """Clean response to ensure pure Hebrew medical text."""
        
        if not response:
            return "לא נמצא מידע רלוונטי במסמכים."
        
        # Aggressively remove all foreign characters
        response = self.foreign_pattern.sub(' ', response)  # Remove Chinese/Japanese
        response = self.mixed_lang_pattern.sub(' ', response)  # Remove any other non-Hebrew
        
        # Remove long sequences of Latin characters (but keep medical abbreviations)
        def replace_latin(match):
            word = match.group()
            # Keep common medical abbreviations
            medical_abbrevs = ['MPJ', 'NV', 'ICD', 'K-WIRE', 'FRACTURE', 'METACARPALS', 'CIL', 'SAMMY', 'HABIB']
            if word.upper() in medical_abbrevs:
                return word
            return ' '
        
        response = re.sub(r'[a-zA-Z]{2,}', replace_latin, response)
        
        # Remove any remaining suspicious patterns
        response = re.sub(r'如下信息.*?：', '', response)  # Remove Chinese instructions
        response = re.sub(r'根据.*?：', '', response)  # Remove Chinese patterns
        response = re.sub(r'将被转换.*?：', '', response)  # Remove conversion instructions
        response = re.sub(r'可以根据.*?：', '', response)  # Remove Chinese analysis
        
        # Clean up extra whitespace and formatting issues
        response = re.sub(r'\s+', ' ', response)  # Multiple spaces to single space
        response = re.sub(r'\n\s*\n', '\n\n', response)  # Clean paragraph breaks
        response = response.strip()
        
        # Ensure we have Hebrew content
        if not self.hebrew_pattern.search(response) or len(response) < 10:
            return "לא נמצא מידע רלוונטי במסמכים."
        
        return response
    
    def is_response_complete(self, response: str) -> bool:
        """Check if response is complete and coherent."""
        
        if not response or len(response.strip()) < 10:
            return False
        
        response = response.strip()
        
        # Check for incomplete patterns
        for pattern in self.incomplete_patterns:
            if pattern.search(response):
                return False
        
        # Check if ends with proper sentence ending
        if not self.sentence_endings.search(response[-3:]):
            return False
        
        return True
    
    def enhance_response(self, response: str, query: str, sources: List[Dict[str, Any]], model: str = "gemma2:2b-instruct") -> str:
        """Enhance response for completeness and coherence."""
        
        if self.is_response_complete(response):
            return response
        
        # Extract additional context from sources
        additional_context = []
        for source in sources[:3]:
            quote = source.get('quote', '')
            if quote and len(quote) > 50:
                additional_context.append(quote[:500])
        
        context_text = "\n\n".join(additional_context) if additional_context else ""
        
        enhancement_prompt = f"""
השלם ושפר את התשובה הבאה כך שתהיה מלאה, קוהרנטית ומבוססת על המקורות:

שאלה מקורית: {query}

תשובה חסרה: {response}

מקורות נוספים לשיפור:
{context_text}

הוראות:
• ענה בעברית בלבד - אסור לכלול מילים באנגלית או סינית
• השלם משפטים חתוכים
• הוסף מידע רלוונטי מהמקורות
• ודא שהתשובה מסתיימת במשפט מלא
• שמור על אותו סגנון ושפה עברית
• אל תוסיף מידע שלא במקורות
• השתמש בתאריכים בפורמט ישראלי (DD/MM/YYYY)

תשובה משופרת בעברית:
"""
        
        try:
            enhanced = gemini_generate(
                enhancement_prompt,
                system="אתה רופא מומחה המשפר תשובות רפואיות בעברית. הפוך את התשובות למלאות, קוהרנטיות ומדויקות על בסיס המקורות בלבד.",
                model="gemini-1.5-flash",
                temperature=0.1,
                max_tokens=1024,
                json_only=False
            )
            
            if isinstance(enhanced, str) and len(enhanced.strip()) > len(response.strip()):
                return enhanced.strip()
        except Exception:
            pass
        
        return response
    
    def add_source_attribution(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Add proper source attribution to response."""
        
        if not sources:
            return response
        
        # Group sources by document type and date
        source_groups = {}
        for source in sources:
            doc_type = source.get('document_type', 'מסמך רפואי')
            doc_date = source.get('document_date', '')
            
            key = f"{doc_type}_{doc_date}"
            if key not in source_groups:
                source_groups[key] = {
                    'doc_type': doc_type,
                    'doc_date': doc_date,
                    'count': 0
                }
            source_groups[key]['count'] += 1
        
        # Create attribution text
        if len(source_groups) > 0:
            source_list = []
            for group in source_groups.values():
                if group['doc_date']:
                    try:
                        formatted_date = datetime.strptime(group['doc_date'], "%Y-%m-%d").strftime("%d/%m/%Y")
                        source_list.append(f"{group['doc_type']} ({formatted_date})")
                    except:
                        source_list.append(f"{group['doc_type']} ({group['doc_date']})")
                else:
                    source_list.append(group['doc_type'])
            
            attribution = f" על פי: {', '.join(source_list)}."
            
            # Add attribution if not already present
            if not any(phrase in response for phrase in ["על פי", "לפי", "בהתאם ל"]):
                response = response.rstrip('.') + attribution
        
        return response


def create_intelligent_rag_response(
    query: str,
    search_function,
    ocr_dir: str,
    model: str = "gemma2:2b-instruct",
    top_k: int = 12
) -> Dict[str, Any]:
    """Create intelligent RAG response with query enhancement and response decoding."""
    
    # Initialize intelligence systems
    query_intelligence = QueryIntelligenceSystem()
    response_decoder = ResponseDecoder()
    
    # Step 1: Analyze query intent
    intent_analysis = query_intelligence.analyze_query_intent(query)
    
    # Step 2: Enhance query based on intent
    enhanced_query = query_intelligence.enhance_query(query, intent_analysis)
    
    # Step 3: Perform enhanced search
    search_results = search_function(
        ocr_dir, 
        enhanced_query, 
        top_k=top_k,
        filters=None
    )
    
    # Step 4: Build enhanced context
    if intent_analysis['month_filter']:
        # Filter results by month for temporal queries
        month_filter = intent_analysis['month_filter']
        filtered_results = []
        for result in search_results:
            doc_date = result.get('document_date', '')
            if doc_date:
                try:
                    result_month = int(doc_date.split('-')[1])
                    if result_month == month_filter:
                        filtered_results.append(result)
                except:
                    pass
        search_results = filtered_results if filtered_results else search_results
    
    # Build context from search results
    context_parts = []
    for i, result in enumerate(search_results[:8], 1):
        doc_type = result.get('document_type', 'מסמך')
        doc_date = result.get('document_date', '')
        text = result.get('text', '')[:1000]
        
        context_parts.append(f"מסמך {i}: {doc_type} ({doc_date})\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Step 5: Create intelligent prompt
    smart_prompt = query_intelligence.create_smart_prompt(enhanced_query, context, intent_analysis)
    
    # Step 6: Generate response using Gemini API
    try:
        raw_response = gemini_generate(
            smart_prompt,
            system="אתה רופא מומחה ישראלי המנתח מסמכים רפואיים בעברית. ענה באופן מקצועי, מפורט ומדויק בעברית בלבד. התבסס אך ורק על המידע במסמכים. אסור לשלב מילים באנגלית, סינית או כל שפה אחרת.",
            model="gemini-1.5-flash",
            temperature=0.2,
            max_tokens=2048,
            json_only=False
        )
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to ollama if Gemini fails
        try:
            print("Falling back to Ollama...")
            raw_response = ollama_generate(
                smart_prompt,
                model="gemma2:2b-instruct",
                system="אתה רופא מומחה ישראלי המנתח מסמכים רפואיים בעברית. ענה באופן מקצועי ומדויק בעברית בלבד.",
                json_only=False
            )
        except Exception as ollama_e:
            print(f"Ollama fallback also failed: {ollama_e}")
            raw_response = "לא ניתן היה לעבד את השאלה. אנא ודא שה-Gemini API key מוגדר נכון או ש-Ollama פועל."
    
    # Step 7: Clean and decode response  
    cleaned_response = response_decoder.clean_hebrew_response(
        raw_response if isinstance(raw_response, str) else str(raw_response)
    )
    
    enhanced_response = response_decoder.enhance_response(
        cleaned_response,
        query,
        search_results[:3],
        model
    )
    
    # Step 8: Add source attribution
    final_response = response_decoder.add_source_attribution(enhanced_response, search_results[:3])
    
    # Step 9: Build sources for API
    sources = []
    seen_ids = set()
    for result in search_results[:4]:
        doc_id = str(result.get('document_id', ''))
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            sources.append({
                'document_id': doc_id,
                'document_type': result.get('document_type', ''),
                'document_date': result.get('document_date', ''),
                'pages': result.get('pages', []),
                'quote': result.get('text', '')[:800]
            })
    
    return {
        'question': query,
        'enhanced_query': enhanced_query,
        'answer': final_response,
        'sources': sources,
        'intent_analysis': intent_analysis
    }


__all__ = [
    'QueryIntelligenceSystem',
    'ResponseDecoder', 
    'create_intelligent_rag_response'
]
