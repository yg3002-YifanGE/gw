#!/usr/bin/env python3
"""
Quick script to check if models are cached locally
"""
import sys
import os

print("Checking Transformers Cache...", flush=True)
print("="*60, flush=True)

try:
    from transformers import file_utils
    cache_dir = file_utils.default_cache_path
    print(f"Cache directory: {cache_dir}", flush=True)
    print(f"Exists: {os.path.exists(cache_dir)}", flush=True)
    
    if os.path.exists(cache_dir):
        # Check for distilbert
        distilbert_patterns = [
            "models--distilbert-base-uncased",
            "distilbert-base-uncased"
        ]
        
        found = False
        for root, dirs, files in os.walk(cache_dir):
            for pattern in distilbert_patterns:
                if pattern in root:
                    print(f"\n✓ Found DistilBERT cache:", flush=True)
                    print(f"  {root}", flush=True)
                    found = True
                    # Check for tokenizer files
                    tokenizer_files = [f for f in files if 'tokenizer' in f.lower() or 'vocab' in f.lower()]
                    if tokenizer_files:
                        print(f"  Tokenizer files: {len(tokenizer_files)}", flush=True)
                    break
            if found:
                break
        
        if not found:
            print("\n⚠ DistilBERT not found in cache", flush=True)
            print("  It will be downloaded on first use (may take 5-15 minutes)", flush=True)
    
    # List all cached models
    print("\nCached models:", flush=True)
    if os.path.exists(cache_dir):
        items = os.listdir(cache_dir)
        model_dirs = [item for item in items if item.startswith("models--")]
        if model_dirs:
            for model_dir in model_dirs[:10]:  # Show first 10
                print(f"  - {model_dir.replace('models--', '')}", flush=True)
            if len(model_dirs) > 10:
                print(f"  ... and {len(model_dirs) - 10} more", flush=True)
        else:
            print("  (none)", flush=True)
    
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n" + "="*60, flush=True)
print("Tip: If models are not cached, they will download automatically", flush=True)
print("     This can take 5-15 minutes on first run.", flush=True)

