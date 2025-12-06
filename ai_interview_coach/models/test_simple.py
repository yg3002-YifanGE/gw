#!/usr/bin/env python3
"""
Simple step-by-step test to find where it gets stuck
"""
import sys
import os

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

def test_step(step_num, step_name, func):
    """Run a test step with clear output"""
    print(f"\n{'='*60}", flush=True)
    print(f"STEP {step_num}: {step_name}", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        result = func()
        print(f"✓ STEP {step_num} completed successfully", flush=True)
        return result
    except KeyboardInterrupt:
        print(f"\n✗ STEP {step_num} interrupted by user (Ctrl+C)", flush=True)
        print(f"This is where the script got stuck!", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"✗ STEP {step_num} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Step 1: Basic imports
def step1():
    print("Importing torch...", flush=True)
    import torch
    print(f"  PyTorch {torch.__version__}", flush=True)
    print(f"  CUDA: {torch.cuda.is_available()}", flush=True)
    return torch

torch = test_step(1, "Import PyTorch", step1)

# Step 2: Import transformers
def step2():
    print("Importing transformers...", flush=True)
    import transformers
    print(f"  Transformers {transformers.__version__}", flush=True)
    return transformers

transformers = test_step(2, "Import Transformers", step2)

# Step 3: Check files
def step3():
    print("Checking required files...", flush=True)
    checkpoint = "../checkpoints/stage2/best_model.pt"
    data = "../data/training_data/interview_answers_annotated.json"
    
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"  ✓ Checkpoint: {os.path.getsize(checkpoint)/1024/1024:.1f} MB", flush=True)
    
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found: {data}")
    print(f"  ✓ Data file exists", flush=True)
    return checkpoint, data

checkpoint_path, data_path = test_step(3, "Check Files", step3)

# Step 4: Import tokenizer class
def step4():
    print("Importing BertTokenizer class...", flush=True)
    from transformers import BertTokenizer
    print("  ✓ BertTokenizer imported", flush=True)
    return BertTokenizer

BertTokenizer = test_step(4, "Import Tokenizer Class", step4)

# Step 5: Load tokenizer (THIS IS WHERE IT MIGHT GET STUCK)
def step5():
    print("Loading tokenizer 'distilbert-base-uncased'...", flush=True)
    print("  ⚠ This may download ~50MB if first time", flush=True)
    print("  ⚠ Please wait, this can take 1-5 minutes...", flush=True)
    import time
    start = time.time()
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    elapsed = time.time() - start
    print(f"  ✓ Tokenizer loaded in {elapsed:.1f}s", flush=True)
    return tokenizer

tokenizer = test_step(5, "Load Tokenizer", step5)

# Step 6: Import model class
def step6():
    print("Importing BERTAnswerScorer...", flush=True)
    from answer_scorer import BERTAnswerScorer
    print("  ✓ BERTAnswerScorer imported", flush=True)
    return BERTAnswerScorer

BERTAnswerScorer = test_step(6, "Import Model Class", step6)

# Step 7: Initialize model (THIS IS WHERE IT MIGHT GET STUCK)
def step7():
    print("Initializing BERTAnswerScorer...", flush=True)
    print("  ⚠ This may download ~250MB DistilBERT model if first time", flush=True)
    print("  ⚠ Please wait, this can take 5-15 minutes...", flush=True)
    import time
    start = time.time()
    model = BERTAnswerScorer()
    elapsed = time.time() - start
    print(f"  ✓ Model initialized in {elapsed:.1f}s", flush=True)
    return model

model = test_step(7, "Initialize Model", step7)

# Step 8: Load checkpoint
def step8():
    print("Loading checkpoint (760MB)...", flush=True)
    print("  ⚠ This may take 10-30 seconds...", flush=True)
    import time
    start = time.time()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    elapsed = time.time() - start
    print(f"  ✓ Checkpoint loaded in {elapsed:.1f}s", flush=True)
    print(f"  Keys: {list(checkpoint.keys())}", flush=True)
    
    print("Loading state dict...", flush=True)
    start = time.time()
    model.load_state_dict(checkpoint['model_state_dict'])
    elapsed = time.time() - start
    print(f"  ✓ State dict loaded in {elapsed:.1f}s", flush=True)
    return checkpoint

checkpoint = test_step(8, "Load Checkpoint", step8)

# Step 9: Load data
def step9():
    print("Creating data loaders...", flush=True)
    from data_loader import create_data_loaders
    loaders = create_data_loaders(
        interview_path=data_path,
        batch_size=8
    )
    print(f"  ✓ Data loaders created", flush=True)
    print(f"  Test batches: {len(loaders['test'])}", flush=True)
    return loaders

loaders = test_step(9, "Load Data", step9)

print("\n" + "="*60, flush=True)
print("ALL TESTS PASSED!", flush=True)
print("="*60, flush=True)
print("\nThe evaluation script should work now.", flush=True)
print("Run: python -u evaluate.py --checkpoint ../checkpoints/stage2/best_model.pt --data_path ../data/training_data/interview_answers_annotated.json --data_type interview --output_dir ../evaluation_results", flush=True)

