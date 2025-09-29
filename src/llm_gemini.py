"""
Google Gemini API integration for Hebrew medical RAG system.
Now uses the quota-aware system from robust_zpe for intelligent rate limiting,
API key rotation, and model fallback.
"""

# Import from the new quota-aware implementation
try:
    from .llm_gemini_quota_aware import (
        gemini_generate,
        test_gemini_api,
        get_gemini_client,
        QuotaAwareGeminiClient
    )
except ImportError:
    # Fallback to basic implementation for backward compatibility
    import json
    import os
    from typing import Dict, Any, Optional
    import requests

    def gemini_generate(
        prompt: str,
        system: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_only: bool = False,
    ) -> str | Dict[str, Any]:
        """Fallback implementation without quota management."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        if json_only:
            full_prompt += "\n\nהחזר את התשובה בפורמט JSON בלבד."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        if "candidates" not in data or not data["candidates"]:
            raise Exception("No response candidates from Gemini API")
        
        generated_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        if json_only:
            try:
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_text = generated_text[start_idx:end_idx + 1]
                    return json.loads(json_text)
                return json.loads(generated_text)
            except json.JSONDecodeError:
                return {"answer": generated_text}
        
        return generated_text

    def test_gemini_api() -> bool:
        """Test if Gemini API is working correctly."""
        try:
            response = gemini_generate("Test", temperature=0.1)
            return isinstance(response, str) and len(response) > 0
        except Exception:
            return False


__all__ = [
    "gemini_generate",
    "test_gemini_api",
    "get_gemini_client",
    "QuotaAwareGeminiClient"
]
