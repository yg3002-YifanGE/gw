# Model Training Guide

Complete guide for training the answer scoring model with transfer learning.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Data Preparation](#data-preparation)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Integration](#integration)

---

## Overview

**Training Strategy: Transfer Learning + Domain Adaptation**

- **Stage 1**: Pre-train on ASAP-AES dataset (12,978 essays)
- **Stage 2**: Fine-tune on annotated interview Q&As (50-100 samples)

**Model Architecture**: DistilBERT + Multi-head Scoring Layer

**Scoring Dimensions**:
- Content Relevance (35%)
- Technical Accuracy (35%)
- Communication Clarity (15%)
- STAR Structure (15%)

---

## Setup

### 1. Install Dependencies

```bash
cd ai_interview_coach
pip install torch transformers scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Verify Installation

```bash
python -c "import torch; import transformers; print('✓ Setup complete')"
```

---

## Data Preparation

### Step 1: Download ASAP Dataset

1. Go to: https://www.kaggle.com/c/asap-aes/data
2. Download `training_set_rel3.tsv`
3. Save to: `data/training_data/training_set_rel3.tsv`

OR use the automated script:

```bash
cd scripts
bash prepare_asap_data.sh
```

**Expected output**: `data/training_data/asap_essays.csv` (~9,000 essays from sets 2-6)

### Step 2: Generate Interview Data for Annotation

```bash
cd scripts
python generate_interview_data.py
```

**Output**: 
- `data/training_data/interview_answers_to_annotate.json` (100 Q&A pairs)
- `data/training_data/annotation_template.txt` (guidelines)

### Step 3: Manual Annotation

**Time required**: 2-3 hours (can split among team members)

1. Open `interview_answers_to_annotate.json`
2. For each Q&A pair, adjust the scores (1-5):
   - `overall_score`: Overall quality
   - `breakdown.content_relevance`: Addresses the question
   - `breakdown.technical_accuracy`: Correctness
   - `breakdown.communication_clarity`: Expression quality
   - `breakdown.structure_star`: STAR format presence

3. **Scoring Guidelines**:
   - **5.0**: Excellent - Comprehensive, accurate, STAR structured
   - **4.0**: Good - Solid with minor gaps
   - **3.0**: Average - Basic understanding
   - **2.0**: Below Average - Significant issues
   - **1.0**: Poor - Incorrect/off-topic

4. Save as: `data/training_data/interview_answers_annotated.json`

**Tip**: The generated answers have `[QUALITY_LEVEL]` tags as suggestions. You can edit both scores AND answer text for realism.

### Step 4: Verify Data

```bash
# Check ASAP data
wc -l data/training_data/asap_essays.csv

# Check interview data
python -c "import json; d=json.load(open('data/training_data/interview_answers_annotated.json')); print(f'✓ {len(d)} annotated samples')"
```

---

## Training Pipeline

### Stage 1: Pre-train on ASAP (2-3 hours on GPU)

```bash
cd models
python train.py \
  --stage 1 \
  --asap_path ../data/training_data/asap_essays.csv \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --save_dir ../checkpoints/stage1
```

**Expected results**:
- Val MAE: ~0.4-0.6
- Val Accuracy (±1): ~85-90%

**Outputs**:
- `checkpoints/stage1/best_model.pt`
- `checkpoints/stage1/training_history.json`

### Stage 2: Fine-tune on Interview Data (30-60 minutes)

```bash
python train.py \
  --stage 2 \
  --interview_path ../data/training_data/interview_answers_annotated.json \
  --load_checkpoint ../checkpoints/stage1/best_model.pt \
  --batch_size 8 \
  --epochs 20 \
  --learning_rate 5e-6 \
  --freeze_bert \
  --save_dir ../checkpoints/stage2
```

**Note**: 
- Lower learning rate (5e-6) to avoid catastrophic forgetting
- `--freeze_bert` to only fine-tune the scoring heads
- More epochs (20) due to small dataset

**Expected results**:
- Val MAE: ~0.3-0.5
- Val Accuracy (±1): ~90-95%

**Outputs**:
- `checkpoints/stage2/best_model.pt` (final model)
- `checkpoints/stage2/training_history.json`

---

## Evaluation

### Comprehensive Evaluation

```bash
cd models
python evaluate.py \
  --checkpoint ../checkpoints/stage2/best_model.pt \
  --data_path ../data/training_data/interview_answers_annotated.json \
  --data_type interview \
  --output_dir ../evaluation_results
```

**Outputs**:
- `evaluation_results/evaluation_results.json` (metrics)
- `evaluation_results/evaluation_plots.png` (visualizations)
- `evaluation_results/predictions.json` (sample predictions)

### Key Metrics to Report

- **MAE (Mean Absolute Error)**: Target < 0.5
- **RMSE (Root Mean Square Error)**: Target < 0.7
- **Accuracy (±1 point)**: Target > 90%
- **Correlation**: Target > 0.85

---

## Integration

### Option 1: Standalone API Endpoint

Add to `app/main.py`:

```python
@app.post("/api/evaluate/model", tags=["evaluation"])
def evaluate_with_model(question: str, answer: str):
    """Evaluate answer using trained model"""
    from models.answer_scorer import BERTAnswerScorer
    import torch
    
    # Load model (cache this in production)
    model = BERTAnswerScorer()
    checkpoint = torch.load('checkpoints/stage2/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Predict
    result = model.predict(question, answer)
    
    return {
        "overall_score": result['overall_score'],
        "breakdown": result['breakdown'],
        "method": "bert_model"
    }
```

### Option 2: Hybrid Approach (Model + Heuristic)

Modify `services/eval.py`:

```python
def hybrid_feedback(answer: str, question: str, retrieved: List[Dict]) -> Dict:
    """Combine model and heuristic scoring"""
    
    # Get heuristic scores
    heuristic_result = heuristic_feedback(answer, retrieved)
    
    # Get model scores (if model available)
    try:
        from models.answer_scorer import BERTAnswerScorer
        import torch
        
        model = get_cached_model()  # Load once, cache
        model_result = model.predict(question, answer)
        
        # Blend: 70% model, 30% heuristic
        blended_score = (
            0.7 * model_result['overall_score'] + 
            0.3 * heuristic_result['overall_score']
        )
        
        return {
            'overall_score': blended_score,
            'breakdown': model_result['breakdown'],
            'method': 'hybrid',
            'model_score': model_result['overall_score'],
            'heuristic_score': heuristic_result['overall_score']
        }
    except:
        # Fallback to heuristic
        return heuristic_result
```

---

## Experiments for Report

### Baseline Comparison

1. **Heuristic Only** (current system)
2. **ASAP Pre-trained** (Stage 1 only)
3. **Fine-tuned** (Stage 1 + Stage 2) ✓ Best
4. **Interview-only** (no transfer learning)

Run all variants and compare MAE/RMSE.

### Ablation Study

Test impact of:
- Transfer learning (w/ vs w/o ASAP pre-training)
- Dimension-specific heads (vs single score head)
- Fine-tuning strategy (freeze vs full fine-tune)

### Hyperparameter Tuning

Try different:
- Learning rates: [1e-5, 2e-5, 5e-5]
- Batch sizes: [8, 16, 32]
- Dropout rates: [0.1, 0.3, 0.5]

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**:
```bash
# Reduce batch size
python train.py --batch_size 8  # or 4

# Use gradient accumulation
# (modify train.py to accumulate gradients every 2 steps)
```

### Issue: Model Overfitting

**Symptoms**: Train loss ↓ but Val loss ↑

**Solutions**:
- Increase dropout: `--dropout 0.5`
- Add data augmentation (paraphrase answers)
- Early stopping (already implemented)

### Issue: Poor Performance on Interview Data

**Solutions**:
- Annotate more interview samples (aim for 150-200)
- Use a smaller learning rate in Stage 2
- Don't freeze BERT in Stage 2

### Issue: Slow Training

**Solutions**:
- Use GPU: `--device cuda`
- Use DistilBERT instead of BERT (already default)
- Reduce max_length: modify `max_length=256` in data_loader.py

---

## Quick Reference

### Full Pipeline Commands

```bash
# 1. Prepare data
bash scripts/prepare_asap_data.sh
python scripts/generate_interview_data.py
# (Manual: Annotate interview_answers_to_annotate.json)

# 2. Train Stage 1
python models/train.py --stage 1 \
  --asap_path ../data/training_data/asap_essays.csv \
  --epochs 10 --save_dir ../checkpoints/stage1

# 3. Train Stage 2
python models/train.py --stage 2 \
  --interview_path ../data/training_data/interview_answers_annotated.json \
  --load_checkpoint ../checkpoints/stage1/best_model.pt \
  --epochs 20 --save_dir ../checkpoints/stage2

# 4. Evaluate
python models/evaluate.py \
  --checkpoint ../checkpoints/stage2/best_model.pt \
  --data_path ../data/training_data/interview_answers_annotated.json \
  --data_type interview
```

### File Structure

```
ai_interview_coach/
├── models/
│   ├── answer_scorer.py      # BERT model
│   ├── data_loader.py         # Dataset loaders
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── scripts/
│   ├── generate_interview_data.py  # Data generation
│   └── prepare_asap_data.sh        # ASAP download helper
├── data/
│   └── training_data/
│       ├── asap_essays.csv
│       ├── interview_answers_annotated.json
│       └── annotation_template.txt
├── checkpoints/
│   ├── stage1/best_model.pt
│   └── stage2/best_model.pt
└── TRAINING_README.md
```

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section above
2. Review training logs in `checkpoints/*/training_history.json`
3. Test with smaller dataset first (`--epochs 2`)

---

## Citation

If using ASAP-AES dataset in your report:

```
@inproceedings{asap2012,
  title={A Dataset for Automated Essay Scoring},
  booktitle={Kaggle ASAP-AES Competition},
  year={2012},
  url={https://www.kaggle.com/c/asap-aes}
}
```

Good luck with your training! 🚀

