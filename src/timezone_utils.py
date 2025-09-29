"""
Jerusalem Timezone Utilities for Medical RAG System

This module provides Jerusalem timezone support and current date awareness
for the medical RAG system, ensuring proper date calculations for:
- Sick leave certificate validity
- Medical condition recency assessment
- Document chronological ordering
"""

import os
from datetime import datetime, date
from typing import Optional, Union
import pytz
from tzlocal import get_localzone

# Jerusalem timezone
JERUSALEM_TZ = pytz.timezone('Asia/Jerusalem')

def get_jerusalem_now() -> datetime:
    """Get current datetime in Jerusalem timezone."""
    return datetime.now(JERUSALEM_TZ)

def get_jerusalem_today() -> date:
    """Get current date in Jerusalem timezone."""
    return get_jerusalem_now().date()

def get_jerusalem_time_string() -> str:
    """Get current time string in Jerusalem timezone."""
    now = get_jerusalem_now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def get_jerusalem_date_string() -> str:
    """Get current date string in Jerusalem format (DD/MM/YYYY)."""
    today = get_jerusalem_today()
    return today.strftime("%d/%m/%Y")

def parse_date_with_jerusalem_tz(date_str: str, format_str: str = "%d/%m/%Y") -> Optional[datetime]:
    """Parse date string and return datetime in Jerusalem timezone."""
    try:
        # Parse the date (naive)
        naive_date = datetime.strptime(date_str, format_str)

        # Localize to Jerusalem timezone
        jerusalem_date = JERUSALEM_TZ.localize(naive_date)

        return jerusalem_date
    except (ValueError, TypeError):
        return None

def is_date_current(date_str: str, days_threshold: int = 30) -> bool:
    """Check if a date is within the current threshold (default 30 days)."""
    try:
        doc_date = parse_date_with_jerusalem_tz(date_str)
        if not doc_date:
            return False

        current_date = get_jerusalem_now()
        days_diff = (current_date - doc_date).days

        return abs(days_diff) <= days_threshold
    except Exception:
        return False

def is_date_recent(date_str: str, days_threshold: int = 7) -> bool:
    """Check if a date is very recent (default 7 days)."""
    return is_date_current(date_str, days_threshold)

def format_date_for_display(dt: Union[datetime, date]) -> str:
    """Format datetime/date for display in Hebrew medical context."""
    if isinstance(dt, datetime):
        return dt.strftime("%d/%m/%Y %H:%M")
    elif isinstance(dt, date):
        return dt.strftime("%d/%m/%Y")
    return str(dt)

def get_days_since(date_str: str) -> Optional[int]:
    """Get number of days since the given date."""
    try:
        doc_date = parse_date_with_jerusalem_tz(date_str)
        if not doc_date:
            return None

        current_date = get_jerusalem_now()
        days_diff = (current_date.date() - doc_date.date()).days

        return days_diff
    except Exception:
        return None

def is_sick_leave_valid(end_date_str: str) -> tuple[bool, str]:
    """
    Check if sick leave is still valid based on end date.

    Returns:
        tuple: (is_valid, status_message)
    """
    try:
        end_date = parse_date_with_jerusalem_tz(end_date_str)
        if not end_date:
            return False, "תאריך לא תקין"

        current_date = get_jerusalem_now()

        if current_date.date() <= end_date.date():
            days_left = (end_date.date() - current_date.date()).days
            if days_left == 0:
                return True, "בתוקף עד היום"
            elif days_left == 1:
                return True, "בתוקף עד מחר"
            else:
                return True, f"בתוקף עוד {days_left} ימים"
        else:
            days_expired = (current_date.date() - end_date.date()).days
            return False, f"פג תוקף לפני {days_expired} ימים"

    except Exception as e:
        return False, f"שגיאה בבדיקת תוקף: {str(e)}"

def get_medical_context_relevancy(date_str: str) -> str:
    """
    Get medical context relevancy based on document date.

    Returns contextual description of how recent the medical information is.
    """
    days_since = get_days_since(date_str)

    if days_since is None:
        return "תאריך לא ידוע"

    if days_since == 0:
        return "מהיום"
    elif days_since == 1:
        return "מאתמול"
    elif days_since <= 7:
        return f"מלפני {days_since} ימים"
    elif days_since <= 30:
        weeks = days_since // 7
        return f"מלפני {weeks} שבועות"
    elif days_since <= 365:
        months = days_since // 30
        return f"מלפני {months} חודשים"
    else:
        years = days_since // 365
        return f"מלפני {years} שנים"

# Global current date for the system
CURRENT_DATE_JERUSALEM = get_jerusalem_today()
CURRENT_DATETIME_JERUSALEM = get_jerusalem_now()

def refresh_current_date():
    """Refresh the global current date variables."""
    global CURRENT_DATE_JERUSALEM, CURRENT_DATETIME_JERUSALEM
    CURRENT_DATE_JERUSALEM = get_jerusalem_today()
    CURRENT_DATETIME_JERUSALEM = get_jerusalem_now()

if __name__ == "__main__":
    # Test the timezone utilities
    print("=== Jerusalem Timezone Test ===")
    print(f"Current Jerusalem time: {get_jerusalem_time_string()}")
    print(f"Current Jerusalem date: {get_jerusalem_date_string()}")
    print(f"System timezone: {get_localzone()}")

    # Test sick leave validation
    test_dates = ["01/10/2025", "01/09/2025", "01/08/2025"]
    for test_date in test_dates:
        is_valid, status = is_sick_leave_valid(test_date)
        print(f"Sick leave {test_date}: {status} ({'Valid' if is_valid else 'Invalid'})")

    # Test medical context relevancy
    test_dates = ["06/09/2025", "01/09/2025", "01/08/2025"]
    for test_date in test_dates:
        relevancy = get_medical_context_relevancy(test_date)
        print(f"Medical info {test_date}: {relevancy}")
