#!/usr/bin/env python3
"""
Test script to verify Hebrew bidi processing is working correctly.
"""

import unicodedata
from bidi.algorithm import get_display

def test_bidi_processing():
    """Test that bidi processing works correctly for Hebrew text."""

    # Test Hebrew text
    hebrew_text = "שלום עולם"
    print(f"Original Hebrew: {hebrew_text}")
    print(f"Visual bidi: {get_display(hebrew_text)}")
    print(f"Logical (no change): {hebrew_text}")
    print()

    # Test mixed Hebrew/English
    mixed_text = "Hello שלום World עולם"
    print(f"Mixed text: {mixed_text}")
    print(f"Visual bidi: {get_display(mixed_text)}")
    print()

    # Test with Unicode normalization
    normalized = unicodedata.normalize("NFC", mixed_text)
    print(f"Normalized: {normalized}")
    print(f"Normalized + bidi: {get_display(normalized)}")
    print()

    # Test multiline
    multiline = "שורה ראשונה\nשורה שנייה\nThird line"
    print(f"Multiline:\n{multiline}")
    print(f"Multiline + bidi:\n{get_display(multiline)}")

if __name__ == "__main__":
    test_bidi_processing()