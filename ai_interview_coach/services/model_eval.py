"""
Model-based evaluation service

Integrates trained BERT model with existing evaluation system
"""
import os
import torch
from typing import Dict, Any, List
from pathlib import Path


# Global model cache
_MODEL_CACHE = None


def get_model():
    """Load and cache the trained model"""
    global _MODEL_CACHE
    
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    try:
        # Import model class
        import sys
        model_dir = Path(__file__).parent.parent / 'models'
        sys.path.insert(0, str(model_dir))
        
        from answer_scorer import BERTAnswerScorer
        
        # Load model
        checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'stage2' / 'best_model.pt'
        
        if not checkpoint_path.exists():
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
            return None
        
        model = BERTAnswerScorer()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        _MODEL_CACHE = model
        print(f"✓ Loaded trained model from {checkpoint_path}")
        
        return model
        
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        return None


def model_feedback(question: str, answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate feedback using trained BERT model
    
    Args:
        question: Interview question text
        answer: Candidate's answer text
        retrieved: Retrieved context (for fallback)
        
    Returns:
        Feedback dict compatible with existing system
    """
    model = get_model()
    
    if model is None:
        # Fallback to heuristic
        from .eval import heuristic_feedback
        return heuristic_feedback(answer, retrieved)
    
    try:
        # Get model prediction (output in 1-5 range for compatibility)
        result = model.predict(question, answer, output_range='1-5')
        
        # Convert to expected format
        breakdown = result['breakdown']
        overall_score = result['overall_score']
        
        # Generate textual feedback based on scores
        strengths = []
        improvements = []
        tips = []
        
        # Content relevance
        if breakdown['content_relevance'] >= 4.0:
            strengths.append("Strong content relevance to the question")
        elif breakdown['content_relevance'] < 3.0:
            improvements.append("Increase topical relevance")
            tips.append("Reference key concepts from the question directly")
        
        # Technical accuracy
        if breakdown['technical_accuracy'] >= 4.0:
            strengths.append("Technically accurate explanation")
        elif breakdown['technical_accuracy'] < 3.0:
            improvements.append("Improve technical accuracy")
            tips.append("Add definitions, formulas, or concrete examples")
        
        # Communication
        if breakdown['communication_clarity'] >= 4.0:
            strengths.append("Clear and well-communicated")
        elif breakdown['communication_clarity'] < 3.0:
            improvements.append("Enhance communication clarity")
            tips.append("Use concise paragraphs and structured reasoning")
        
        # STAR structure
        if breakdown['structure_star'] >= 4.0:
            strengths.append("Excellent STAR structure present")
        elif breakdown['structure_star'] < 3.0:
            improvements.append("Add STAR structure: Situation, Task, Action, Result")
            tips.append("Start with context, then your objective, actions, and measurable outcome")
        
        # Default messages if none generated
        if not strengths:
            strengths.append("Demonstrates understanding of the concept")
        
        if not improvements:
            improvements.append("Consider adding more specific examples")
        
        if not tips:
            tips.append("Practice explaining with concrete scenarios from your experience")
        
        return {
            'overall_score': overall_score,
            'breakdown': breakdown,
            'strengths': strengths,
            'improvements': improvements,
            'tips': tips,
            'method': 'bert_model',
            'evidence': {'model_confidence': 'high'}
        }
        
    except Exception as e:
        print(f"Error in model_feedback: {e}")
        # Fallback to heuristic
        from .eval import heuristic_feedback
        return heuristic_feedback(answer, retrieved)


def hybrid_feedback(question: str, answer: str, retrieved: List[Dict[str, Any]], 
                    model_weight: float = 0.7) -> Dict[str, Any]:
    """
    Hybrid approach: Combine model and heuristic scores
    
    Args:
        question: Interview question
        answer: Candidate's answer
        retrieved: Context for heuristic
        model_weight: Weight for model score (0-1)
        
    Returns:
        Blended feedback
    """
    from .eval import heuristic_feedback
    
    # Get both scores
    model_result = model_feedback(question, answer, retrieved)
    heuristic_result = heuristic_feedback(answer, retrieved)
    
    # If model not available, return heuristic
    if model_result.get('method') != 'bert_model':
        return heuristic_result
    
    # Blend scores
    heuristic_weight = 1.0 - model_weight
    
    blended_overall = (
        model_weight * model_result['overall_score'] +
        heuristic_weight * heuristic_result['overall_score']
    )
    
    # Blend breakdown
    blended_breakdown = {}
    for key in model_result['breakdown'].keys():
        model_val = model_result['breakdown'][key]
        heuristic_val = heuristic_result['breakdown'][key]
        blended_breakdown[key] = (
            model_weight * model_val + heuristic_weight * heuristic_val
        )
    
    # Combine strengths/improvements (union)
    combined_strengths = list(set(
        model_result.get('strengths', []) + 
        heuristic_result.get('strengths', [])
    ))
    
    combined_improvements = list(set(
        model_result.get('improvements', []) +
        heuristic_result.get('improvements', [])
    ))
    
    combined_tips = list(set(
        model_result.get('tips', []) +
        heuristic_result.get('tips', [])
    ))
    
    return {
        'overall_score': round(blended_overall, 2),
        'breakdown': {k: round(v, 2) for k, v in blended_breakdown.items()},
        'strengths': combined_strengths[:5],  # Top 5
        'improvements': combined_improvements[:5],
        'tips': combined_tips[:5],
        'method': 'hybrid',
        'model_score': model_result['overall_score'],
        'heuristic_score': heuristic_result['overall_score'],
        'blend_ratio': f"{int(model_weight*100)}% model / {int(heuristic_weight*100)}% heuristic"
    }


# Example: Update main.py to use model-based feedback
# 
# In app/main.py, modify the submit_answer endpoint:
#
# @app.post("/api/session/{session_id}/answer", ...)
# def submit_answer(session_id: str, req: AnswerRequest):
#     ...
#     # Replace this line:
#     # fb = heuristic_feedback(req.answer_text, ctx)
#     
#     # With one of these:
#     
#     # Option 1: Model only
#     from services.model_eval import model_feedback
#     fb = model_feedback(q.question_text, req.answer_text, ctx)
#     
#     # Option 2: Hybrid (recommended)
#     from services.model_eval import hybrid_feedback
#     fb = hybrid_feedback(q.question_text, req.answer_text, ctx, model_weight=0.7)
#     
#     ...

