#!/usr/bin/env python3
"""
Setup script to configure Gemini API key rotation from api_gemini15.txt
Uses the extensive API key collection for optimal quota management.
"""

import os
import sys
from pathlib import Path

def load_api_keys_from_file(filepath: str = "api_gemini15.txt") -> list:
    """Load API keys from the text file."""
    keys_file = Path(filepath)
    if not keys_file.exists():
        print(f"ERROR: {filepath} not found")
        return []
    
    keys = []
    with open(keys_file, 'r') as f:
        for line in f:
            key = line.strip()
            if key and key.startswith('AIza'):
                keys.append(key)
    
    return keys

def setup_quota_environment(keys: list, primary_key_index: int = 0):
    """Setup environment variables for quota-aware Gemini system."""
    
    if not keys:
        print("ERROR: No valid API keys found")
        return False
    
    if primary_key_index >= len(keys):
        primary_key_index = 0
    
    # Set primary key
    primary_key = keys[primary_key_index]
    os.environ['GEMINI_API_KEY'] = primary_key
    print(f"Set GEMINI_API_KEY: {primary_key[:20]}...")
    
    # Set rotation keys (excluding the primary)
    rotation_keys = [key for i, key in enumerate(keys) if i != primary_key_index]
    if rotation_keys:
        keys_str = ','.join(rotation_keys)
        os.environ['GEMINI_API_KEYS'] = keys_str
        print(f"Set GEMINI_API_KEYS: {len(rotation_keys)} rotation keys")
    
    # Set optimal quota configuration for multiple keys
    os.environ['GEMINI_RPS'] = '2'  # Higher rate with multiple keys
    os.environ['GEMINI_RETRIES'] = '3'
    os.environ['GEMINI_MODEL'] = 'gemini-1.5-flash'
    os.environ['GEMINI_ALT_MODELS'] = 'gemini-2.0-flash-exp,gemini-1.5-pro,gemini-1.5-flash-latest'
    os.environ['GEMINI_MAX_TOKENS'] = '4096'
    os.environ['GEMINI_BACKOFF_BASE'] = '0.5'
    os.environ['GEMINI_BACKOFF_MAX'] = '10'
    
    print(f"Configured quota management:")
    print(f"  - Rate limit: 2 requests/second")
    print(f"  - Total keys available: {len(keys)}")
    print(f"  - Retry attempts: 3")
    print(f"  - Model fallback enabled")
    
    return True

def generate_env_script(keys: list, output_file: str = "setup_gemini_env.sh"):
    """Generate a shell script to set up the environment."""
    
    if not keys:
        print("ERROR: No keys to generate script")
        return False
    
    primary_key = keys[0]
    rotation_keys = keys[1:] if len(keys) > 1 else []
    
    script_content = f'''#!/bin/bash
# Gemini API Key Setup Script
# Generated from api_gemini15.txt with {len(keys)} total keys

# Primary API key
export GEMINI_API_KEY="{primary_key}"

# Additional keys for rotation ({len(rotation_keys)} keys)
'''
    
    if rotation_keys:
        keys_str = ','.join(rotation_keys)
        script_content += f'export GEMINI_API_KEYS="{keys_str}"\n'
    
    script_content += '''
# Quota management configuration
export GEMINI_RPS="2"                    # 2 requests per second with key rotation
export GEMINI_RETRIES="3"                # 3 retry attempts
export GEMINI_MODEL="gemini-1.5-flash"   # Preferred model
export GEMINI_ALT_MODELS="gemini-2.0-flash-exp,gemini-1.5-pro,gemini-1.5-flash-latest"
export GEMINI_MAX_TOKENS="4096"          # Max tokens per request
export GEMINI_BACKOFF_BASE="0.5"         # Fast backoff with multiple keys
export GEMINI_BACKOFF_MAX="10"           # Max backoff time

echo "Gemini API environment configured:"
echo "  Primary key: ${GEMINI_API_KEY:0:20}..."
echo "  Rotation keys: ''' + str(len(rotation_keys)) + ''' available"
echo "  Rate limit: 2 RPS with intelligent throttling"
echo "  Model fallback: enabled"
'''
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_file, 0o755)
    
    print(f"Generated environment setup script: {output_file}")
    return True

def test_quota_system():
    """Test the quota-aware system with loaded keys."""
    try:
        from src.llm_gemini import get_gemini_client, test_gemini_api
        
        print("\nTesting quota-aware system:")
        client = get_gemini_client()
        print(f"  - Client initialized successfully")
        print(f"  - Available API keys: {len(client.api_keys._keys)}")
        print(f"  - Available models: {len(client.models._models)}")
        print(f"  - Rate limit interval: {client.quota.min_interval:.2f}s")
        
        # Test basic API functionality
        if test_gemini_api():
            print(f"  - API test: PASSED")
        else:
            print(f"  - API test: FAILED (check keys)")
            
        return True
        
    except Exception as e:
        print(f"  - System test failed: {e}")
        return False

def main():
    """Main setup function."""
    
    print("=== Gemini Quota-Aware System Setup ===")
    
    # Load API keys
    print("\n1. Loading API keys from api_gemini15.txt...")
    keys = load_api_keys_from_file()
    
    if not keys:
        print("FAILED: No valid API keys found")
        sys.exit(1)
    
    print(f"   Loaded {len(keys)} valid API keys")
    
    # Setup environment
    print("\n2. Configuring environment variables...")
    if not setup_quota_environment(keys):
        print("FAILED: Environment setup failed")
        sys.exit(1)
    
    # Generate shell script
    print("\n3. Generating setup script...")
    if not generate_env_script(keys):
        print("FAILED: Script generation failed")
        sys.exit(1)
    
    # Test system
    print("\n4. Testing quota system...")
    test_quota_system()
    
    print("\n=== Setup Complete ===")
    print("\nTo use in your shell:")
    print("  source setup_gemini_env.sh")
    print("\nOr run this script to configure current Python session.")

if __name__ == "__main__":
    main()
