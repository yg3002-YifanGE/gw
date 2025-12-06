#!/usr/bin/env python3
"""
Test script for hybrid scoring functionality
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

def test_hybrid_scoring():
    """Test hybrid scoring with sample question/answer"""
    print("="*60)
    print("Testing Hybrid Scoring")
    print("="*60)
    
    # Sample question and answer
    question = "What is backpropagation?"
    answer = """Backpropagation is the method neural networks use to calculate gradients. 
    In a previous project (Situation), I needed to train a deep learning model for image classification (Task). 
    I implemented backpropagation with gradient clipping to prevent exploding gradients (Action). 
    This improved training stability and reduced loss by 30% (Result)."""
    
    # Mock retrieved context
    retrieved = [
        {"text": "Backpropagation is a key algorithm in neural network training"},
        {"text": "Gradient descent optimization requires computing gradients"},
    ]
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer[:100]}...")
    print("\n" + "-"*60)
    
    # Test 1: Heuristic only
    print("\n[Test 1] Heuristic Scoring")
    try:
        from services.eval import heuristic_feedback
        heuristic_result = heuristic_feedback(answer, retrieved)
        print(f"  Overall Score: {heuristic_result['overall_score']:.2f}")
        print(f"  Breakdown: {heuristic_result['breakdown']}")
        print("  ✓ Heuristic scoring works")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Model only
    print("\n[Test 2] Model Scoring")
    try:
        from services.model_eval import model_feedback
        model_result = model_feedback(question, answer, retrieved)
        print(f"  Overall Score: {model_result['overall_score']:.2f}")
        print(f"  Breakdown: {model_result['breakdown']}")
        print(f"  Method: {model_result.get('method', 'unknown')}")
        if model_result.get('method') == 'bert_model':
            print("  ✓ Model scoring works (using trained BERT)")
        else:
            print("  ⚠ Model not available, fell back to heuristic")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Hybrid scoring
    print("\n[Test 3] Hybrid Scoring (70% model, 30% heuristic)")
    try:
        from services.model_eval import hybrid_feedback
        hybrid_result = hybrid_feedback(question, answer, retrieved, model_weight=0.7)
        print(f"  Overall Score: {hybrid_result['overall_score']:.2f}")
        print(f"  Breakdown: {hybrid_result['breakdown']}")
        print(f"  Method: {hybrid_result.get('method', 'unknown')}")
        if 'model_score' in hybrid_result:
            print(f"  Model Score: {hybrid_result['model_score']:.2f}")
            print(f"  Heuristic Score: {hybrid_result['heuristic_score']:.2f}")
            print(f"  Blend Ratio: {hybrid_result.get('blend_ratio', 'N/A')}")
        print("  ✓ Hybrid scoring works")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Different weights
    print("\n[Test 4] Hybrid Scoring with different weights")
    try:
        weights = [0.3, 0.5, 0.7, 0.9]
        for weight in weights:
            result = hybrid_feedback(question, answer, retrieved, model_weight=weight)
            print(f"  Weight {weight:.1f}: Score = {result['overall_score']:.2f}")
        print("  ✓ Different weights work correctly")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nYour application is now using hybrid scoring by default.")
    print("You can configure it via SessionConfig:")
    print("  - scoring_method: 'heuristic', 'model', or 'hybrid'")
    print("  - model_weight: 0.0-1.0 (only used in hybrid mode)")
    return True


if __name__ == '__main__':
    success = test_hybrid_scoring()
    sys.exit(0 if success else 1)

