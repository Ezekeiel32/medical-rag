#!/usr/bin/env python3
"""
Quota-Aware Google Gemini API integration for Hebrew medical RAG system.
Based on the robust_zpe quota management system - NO hardcoded responses.
Provides intelligent rate limiting, API key rotation, and model fallback.
"""

import os
import re
import json
import time
import random
import hashlib
import logging
import sqlite3
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    force=True,
)
logger = logging.getLogger(__name__)


def _parse_json_any(text: str) -> Any:
    """Parse first JSON object or array from text, with fallbacks."""
    if not text:
        raise ValueError("Empty response")
    content = text.strip()
    # Strip fenced code blocks if present
    content = re.sub(r"^```[a-zA-Z]*\n", "", content)
    content = re.sub(r"\n```$", "", content)
    # Try full parse first
    try:
        return json.loads(content)
    except Exception:
        pass
    # Try to find an object or array
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
    if m:
        frag = m.group(1)
        try:
            return json.loads(frag)
        except Exception:
            normalized = frag.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            return eval(normalized)
    # Last resort: python literal of whole content
    normalized = content.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    return eval(normalized)


class QuotaManager:
    """Simple quota/rate limiter with per-request throttle and backoff hints."""

    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / max(0.001, float(requests_per_second)) if requests_per_second > 0 else 0.0
        self._last_request_ts = 0.0

    def throttle(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_ts = time.monotonic()


class ApiKeyRotator:
    """Rotate across multiple API keys if provided (GEMINI_API_KEYS comma-separated)."""

    def __init__(self, primary_key: Optional[str] = None, extra_keys: Optional[str] = None):
        keys: List[str] = []
        if extra_keys:
            keys.extend([k.strip() for k in extra_keys.split(',') if k.strip()])
        if primary_key and primary_key.strip() and primary_key not in keys:
            keys.insert(0, primary_key.strip())
        self._keys = keys if keys else ([primary_key] if primary_key else [])
        self._idx = 0

    def current_key(self) -> Optional[str]:
        if not self._keys:
            return None
        return self._keys[self._idx]

    def rotate(self) -> Optional[str]:
        if not self._keys:
            return None
        self._idx = (self._idx + 1) % len(self._keys)
        return self._keys[self._idx]


class ModelRotator:
    """Rotate across Gemini models when endpoints 404/429."""

    def __init__(self, preferred: str, alternatives: List[str]):
        models: List[str] = []
        if preferred:
            models.append(preferred)
        for m in alternatives:
            m = m.strip()
            if m and m not in models:
                models.append(m)
        self._models = models
        self._idx = 0

    def current(self) -> str:
        return self._models[self._idx]

    def rotate(self) -> str:
        self._idx = (self._idx + 1) % len(self._models)
        return self._models[self._idx]


class QuotaAwareGeminiClient:
    """Quota-aware Gemini client with rate limiting, key rotation, and model fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Google API key required! Set GEMINI_API_KEY environment variable")
        
        self.gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.cache_db = "gemini_quota_cache.db"
        self.session = requests.Session()
        self.req_timeout = int(os.getenv('GEMINI_TIMEOUT_SECONDS', '30'))
        self.connect_timeout = int(os.getenv('GEMINI_CONNECT_TIMEOUT', '10'))
        self.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '2048'))
        
        # Quota / rate limiting
        self.quota = QuotaManager(requests_per_second=float(os.getenv('GEMINI_RPS', '1')))
        
        # API key rotation support
        self.api_keys = ApiKeyRotator(
            primary_key=self.api_key,
            extra_keys=os.getenv('GEMINI_API_KEYS')
        )
        
        # Model rotation support
        preferred_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        alt_models_env = os.getenv('GEMINI_ALT_MODELS', 'gemini-2.0-flash-exp,gemini-1.5-pro')
        alt_models = [m.strip() for m in alt_models_env.split(',') if m.strip()]
        self.models = ModelRotator(preferred=preferred_model, alternatives=alt_models)
        
        self.max_retries = int(os.getenv('GEMINI_RETRIES', '2'))
        self.backoff_base = float(os.getenv('GEMINI_BACKOFF_BASE', '1'))
        self.backoff_max = float(os.getenv('GEMINI_BACKOFF_MAX', '20'))
        self.token_step_factor = float(os.getenv('GEMINI_TOKEN_STEP', '0.5'))  # shrink tokens on retry
        
        self._init_cache_db()
        logger.info("Quota-aware Gemini client initialized - Real AI analysis with rate limiting")
    
    def _init_cache_db(self):
        """Initialize Gemini response cache"""
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gemini_cache (
                query_hash TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        json_only: bool = False,
        use_cache: bool = True
    ) -> str | Dict[str, Any]:
        """
        Generate response using Google Gemini API with quota management.
        
        Args:
            prompt: The user prompt
            system: System instructions (optional)
            model: Gemini model to use (overrides default)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
            json_only: Whether to expect JSON response
            use_cache: Whether to use caching
        
        Returns:
            Generated text or parsed JSON dict
        """
        # Override model if specified
        if model:
            original_model = self.models._models[0]
            self.models._models[0] = model
            
        # Construct the full prompt with system instructions
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        # Add JSON instruction if needed
        if json_only:
            full_prompt += "\n\nReturn the response in JSON format only."
        
        # Use provided max_tokens or default
        current_max_tokens = max_tokens or self.max_tokens
        
        try:
            response_text = self._query_gemini(
                full_prompt, 
                temperature=temperature, 
                max_tokens=current_max_tokens,
                use_cache=use_cache
            )
            
            # Parse JSON if requested
            if json_only:
                try:
                    # Try to find JSON in the response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_text = response_text[start_idx:end_idx + 1]
                        return json.loads(json_text)
                    else:
                        # If no JSON found, try parsing the whole response
                        return json.loads(response_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return a default structure
                    return {"answer": response_text}
            
            return response_text
            
        finally:
            # Restore original model if it was overridden
            if model and 'original_model' in locals():
                self.models._models[0] = original_model
    
    def _query_gemini(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048, use_cache: bool = True) -> str:
        """Query Gemini AI with quota management and caching"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check cache
        if use_cache:
            conn = sqlite3.connect(self.cache_db)
            cached = conn.execute(
                "SELECT response FROM gemini_cache WHERE query_hash = ?", 
                (prompt_hash,)
            ).fetchone()
            
            if cached:
                conn.close()
                logger.info("Using cached Gemini response")
                return cached[0]
            conn.close()
        
        logger.info("Querying Gemini AI with quota management...")
        
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": max_tokens,
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # Attempt loop with quota-aware throttle, backoff, and rotation
            current_tokens = max_tokens
            for attempt in range(self.max_retries + 1):
                self.quota.throttle()
                model_name = self.models.current()
                api_key = self.api_keys.current_key() or self.api_key
                url = f"{self.gemini_base_url}/{model_name}:generateContent?key={api_key}"
                
                logger.info(
                    f"[Gemini] attempt={attempt+1}/{self.max_retries+1} model={model_name} tokens={int(current_tokens)} timeout={self.req_timeout}s"
                )
                
                payload["generationConfig"]["maxOutputTokens"] = int(current_tokens)
                t0 = time.monotonic()
                
                try:
                    response = self.session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=(self.connect_timeout, self.req_timeout)
                    )
                    
                    if response.status_code == 404:
                        logger.warning("[Gemini] 404 Not Found - rotating model and retrying")
                        self.models.rotate()
                        continue
                        
                    if response.status_code == 429:
                        retry_after = response.headers.get('Retry-After')
                        delay = float(retry_after) if retry_after else min(self.backoff_base * (2 ** attempt), self.backoff_max)
                        jitter = random.random()
                        logger.warning(f"[Gemini] 429 Too Many Requests - delaying {delay + jitter:.1f}s and rotating key/model")
                        time.sleep(delay + jitter)
                        
                        # Rotate API key if available; also rotate model if no alternate key or key unchanged
                        prev_key = api_key
                        new_key = self.api_keys.rotate()
                        if (not new_key) or (new_key == prev_key):
                            self.models.rotate()
                        current_tokens = max(128, int(current_tokens * self.token_step_factor))
                        continue
                        
                    response.raise_for_status()
                    
                except requests.RequestException as e:
                    # Network/server error backoff
                    delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
                    logger.warning(f"[Gemini] request error: {e} - backoff {delay:.1f}s")
                    time.sleep(delay)
                    continue

                result = response.json()
                logger.info(f"[Gemini] response_ok in {time.monotonic()-t0:.1f}s")
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if "content" not in candidate or "parts" not in candidate["content"]:
                        raise Exception("Invalid response structure from Gemini API")
                        
                    gemini_response = candidate['content']['parts'][0]['text']
                    
                    # Cache the response
                    if use_cache:
                        conn = sqlite3.connect(self.cache_db)
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO gemini_cache 
                            (query_hash, prompt, response, timestamp) 
                            VALUES (?, ?, ?, ?)
                            """,
                            (prompt_hash, prompt, gemini_response, datetime.now().isoformat())
                        )
                        conn.commit()
                        conn.close()
                    
                    return gemini_response
                else:
                    # If empty, try model rotation next
                    logger.warning("[Gemini] Empty candidates - rotating model")
                    self.models.rotate()
                    
            raise Exception("Exhausted retries without successful Gemini response")
                
        except Exception as e:
            logger.error(f"Gemini query failed: {e}")
            raise


# Global client instance
_gemini_client = None

def get_gemini_client() -> QuotaAwareGeminiClient:
    """Get or create the global Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = QuotaAwareGeminiClient()
    return _gemini_client


def gemini_generate(
    prompt: str,
    system: Optional[str] = None,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    json_only: bool = False,
) -> str | Dict[str, Any]:
    """
    Generate response using quota-aware Google Gemini API.
    
    Args:
        prompt: The user prompt
        system: System instructions (optional)
        model: Gemini model to use
        temperature: Response creativity (0.0-1.0)
        max_tokens: Maximum response length
        json_only: Whether to expect JSON response
    
    Returns:
        Generated text or parsed JSON dict
    """
    client = get_gemini_client()
    return client.generate(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_only=json_only
    )


def test_gemini_api() -> bool:
    """Test if quota-aware Gemini API is working correctly."""
    try:
        response = gemini_generate(
            "Hello, what is your name?",
            system="You are a helpful assistant that speaks Hebrew.",
            temperature=0.1
        )
        return isinstance(response, str) and len(response) > 0
    except Exception as e:
        logger.error(f"Gemini API test failed: {e}")
        return False


__all__ = [
    "QuotaAwareGeminiClient",
    "gemini_generate", 
    "test_gemini_api",
    "get_gemini_client"
]
