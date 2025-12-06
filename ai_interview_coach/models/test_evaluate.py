#!/usr/bin/env python3
"""
Quick test script to diagnose evaluation issues
"""
import sys
import os

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("="*60, flush=True)
print("Evaluation Diagnostic Test", flush=True)
print("="*60, flush=True)

# Test 1: Check imports
print("\n[Test 1] Checking imports...", flush=True)
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}", flush=True)
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"  ✗ Error importing torch: {e}", flush=True)
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ Transformers version: {transformers.__version__}", flush=True)
except Exception as e:
    print(f"  ✗ Error importing transformers: {e}", flush=True)
    sys.exit(1)

# Test 2: Check checkpoint file
print("\n[Test 2] Checking checkpoint file...", flush=True)
checkpoint_path = "../checkpoints/stage2/best_model.pt"
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  ✓ Checkpoint exists: {checkpoint_path}", flush=True)
    print(f"  ✓ File size: {size_mb:.1f} MB", flush=True)
else:
    print(f"  ✗ Checkpoint not found: {checkpoint_path}", flush=True)
    print(f"  Current directory: {os.getcwd()}", flush=True)
    sys.exit(1)

# Test 3: Check data file
print("\n[Test 3] Checking data file...", flush=True)
data_path = "../data/training_data/interview_answers_annotated.json"
if os.path.exists(data_path):
    print(f"  - Reading JSON file...", flush=True)
    import json
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ Data file exists: {data_path}", flush=True)
    print(f"  ✓ Number of samples: {len(data)}", flush=True)
else:
    print(f"  ✗ Data file not found: {data_path}", flush=True)
    print(f"  Current directory: {os.getcwd()}", flush=True)
    sys.exit(1)

# Test 3.5: Check transformers cache
print("\n[Test 3.5] Checking transformers cache...", flush=True)
try:
    from transformers import file_utils
    cache_dir = file_utils.default_cache_path
    print(f"  - Transformers cache directory: {cache_dir}", flush=True)
    if os.path.exists(cache_dir):
        print(f"  ✓ Cache directory exists", flush=True)
        # Check if distilbert is cached
        distilbert_path = os.path.join(cache_dir, "models--distilbert-base-uncased")
        if os.path.exists(distilbert_path):
            print(f"  ✓ DistilBERT appears to be cached", flush=True)
        else:
            print(f"  ⚠ DistilBERT not found in cache (will download)", flush=True)
    else:
        print(f"  ⚠ Cache directory does not exist yet", flush=True)
except Exception as e:
    print(f"  ⚠ Could not check cache: {e}", flush=True)

# Test 4: Try loading tokenizer
print("\n[Test 4] Testing tokenizer loading...", flush=True)
print("  - This may take a while if downloading for the first time...", flush=True)
print("  - Please wait...", flush=True)
try:
    from transformers import BertTokenizer
    import time
    start_time = time.time()
    print("  - Calling BertTokenizer.from_pretrained()...", flush=True)
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    elapsed = time.time() - start_time
    print(f"  ✓ Tokenizer loaded successfully (took {elapsed:.1f}s)", flush=True)
except KeyboardInterrupt:
    print("\n  ✗ Interrupted by user", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Error loading tokenizer: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try loading model class
print("\n[Test 5] Testing model class import...", flush=True)
try:
    from answer_scorer import BERTAnswerScorer
    print("  ✓ BERTAnswerScorer imported successfully", flush=True)
except Exception as e:
    print(f"  ✗ Error importing BERTAnswerScorer: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Try initializing model
print("\n[Test 6] Testing model initialization...", flush=True)
print("  - This may take a while if downloading DistilBERT for the first time...", flush=True)
print("  - Please wait...", flush=True)
try:
    import time
    start_time = time.time()
    print("  - Creating model instance...", flush=True)
    model = BERTAnswerScorer()
    elapsed = time.time() - start_time
    print(f"  ✓ Model initialized successfully (took {elapsed:.1f}s)", flush=True)
except KeyboardInterrupt:
    print("\n  ✗ Interrupted by user", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Error initializing model: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Try loading checkpoint
print("\n[Test 7] Testing checkpoint loading...", flush=True)
print("  - Loading 760MB checkpoint file (this may take 10-30 seconds)...", flush=True)
try:
    import time
    start_time = time.time()
    print("  - Reading checkpoint file from disk...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    elapsed = time.time() - start_time
    print(f"  ✓ Checkpoint loaded (took {elapsed:.1f}s)", flush=True)
    print(f"  - Checkpoint keys: {list(checkpoint.keys())}", flush=True)
    
    print("  - Loading state dict into model...", flush=True)
    start_time = time.time()
    model.load_state_dict(checkpoint['model_state_dict'])
    elapsed = time.time() - start_time
    print(f"  ✓ State dict loaded successfully (took {elapsed:.1f}s)", flush=True)
except Exception as e:
    print(f"  ✗ Error loading checkpoint: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Try loading data
print("\n[Test 8] Testing data loading...", flush=True)
try:
    from data_loader import create_data_loaders
    print("  - Creating data loaders...", flush=True)
    loaders = create_data_loaders(
        interview_path=data_path,
        batch_size=8  # Small batch for testing
    )
    test_loader = loaders['test']
    print(f"  ✓ Data loaders created", flush=True)
    print(f"  - Test batches: {len(test_loader)}", flush=True)
except Exception as e:
    print(f"  ✗ Error loading data: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60, flush=True)
print("All tests passed! Evaluation should work now.", flush=True)
print("="*60, flush=True)
print("\nTo run evaluation, use:", flush=True)
print("  python -u evaluate.py \\", flush=True)
print("    --checkpoint ../checkpoints/stage2/best_model.pt \\", flush=True)
print("    --data_path ../data/training_data/interview_answers_annotated.json \\", flush=True)
print("    --data_type interview \\", flush=True)
print("    --output_dir ../evaluation_results", flush=True)
print("\nNote: Use 'python -u' to ensure unbuffered output", flush=True)

