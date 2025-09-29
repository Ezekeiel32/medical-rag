# Quota-Aware Gemini Configuration

The medical RAG system now uses an intelligent quota-aware Gemini integration based on the robust_zpe system. This provides advanced rate limiting, API key rotation, and model fallback capabilities.

## 🚀 **Quick Setup with 70 API Keys**

The system is pre-configured with 70 Gemini API keys from `api_gemini15.txt` for maximum quota capacity:

```bash
# Automatic setup (recommended)
source venv/bin/activate
python setup_quota_keys.py

# Or use generated environment script
source setup_gemini_env.sh
```

This configuration provides:
- **70 API keys** for intelligent rotation
- **2 requests/second** rate limit (optimized for multiple keys)  
- **Automatic failover** across all keys
- **4 model variants** for fallback
- **Enterprise-grade reliability**

## Environment Variables

### Basic Configuration
- `GEMINI_API_KEY` - Primary Gemini API key (required)
- `GEMINI_API_KEYS` - Comma-separated list of additional API keys for rotation (optional)

### Rate Limiting
- `GEMINI_RPS` - Requests per second limit (default: 1)
- `GEMINI_TIMEOUT_SECONDS` - Request timeout in seconds (default: 30)
- `GEMINI_CONNECT_TIMEOUT` - Connection timeout in seconds (default: 10)

### Model Configuration
- `GEMINI_MODEL` - Preferred model (default: "gemini-1.5-flash")
- `GEMINI_ALT_MODELS` - Comma-separated fallback models (default: "gemini-2.0-flash-exp,gemini-1.5-pro")
- `GEMINI_MAX_TOKENS` - Maximum tokens per request (default: 2048)

### Retry and Backoff
- `GEMINI_RETRIES` - Maximum retry attempts (default: 2)
- `GEMINI_BACKOFF_BASE` - Base backoff time in seconds (default: 1)
- `GEMINI_BACKOFF_MAX` - Maximum backoff time in seconds (default: 20)
- `GEMINI_TOKEN_STEP` - Token reduction factor on retry (default: 0.5)

## Features

### Smart Rate Limiting
- Automatic throttling based on configured requests per second
- Intelligent backoff on 429 (Too Many Requests) errors
- Jitter added to prevent thundering herd problems

### API Key Rotation
- Automatic rotation through multiple API keys when rate limits are hit
- Seamless fallback between keys
- Support for primary key plus additional keys

### Model Fallback
- Automatic rotation through different Gemini models on errors
- 404 error handling with model switching
- Token reduction on retries to handle context limits

### Caching
- SQLite-based response caching to reduce API calls
- MD5-based cache keys for efficient lookups
- Automatic cache management

## Example Configuration

```bash
# Basic setup
export GEMINI_API_KEY="your-primary-key-here"

# Advanced setup with rotation and rate limiting
export GEMINI_API_KEY="primary-key"
export GEMINI_API_KEYS="backup-key-1,backup-key-2,backup-key-3"
export GEMINI_RPS="2"
export GEMINI_MODEL="gemini-1.5-flash"
export GEMINI_ALT_MODELS="gemini-2.0-flash-exp,gemini-1.5-pro,gemini-1.5-flash-latest"
export GEMINI_MAX_TOKENS="4096"
export GEMINI_RETRIES="3"
```

## Backward Compatibility

The system maintains full backward compatibility with existing code. All existing calls to `gemini_generate()` will work without modification while gaining the benefits of quota management.

## Monitoring

The system provides detailed logging for:
- Rate limiting actions
- API key rotations
- Model fallbacks
- Cache usage
- Error handling and retries

Check your application logs for `[Gemini]` prefixed messages to monitor quota management behavior.
