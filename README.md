# AI Interview Coach

An intelligent interview practice system that provides automated feedback on technical interview answers using a hybrid scoring approach combining deep learning models and rule-based heuristics.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Data Preparation](#data-preparation)
5. [Model Architecture & Training](#model-architecture--training)
6. [Training Results](#training-results)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)
9. [API Documentation](#api-documentation)
10. [Limitations & Future Work](#limitations--future-work)
11. [Project Structure](#project-structure)
12. [Contributing](#contributing)

---

## Overview

The AI Interview Coach is a comprehensive system designed to help candidates practice technical interviews. It combines:

- **RAG-based Question Retrieval**: Intelligent question selection from a curated database using TF-IDF and cosine similarity
- **Hybrid Scoring System**: Combines BERT-based deep learning model (70%) with rule-based heuristics (30%) for robust and interpretable feedback
- **Multi-dimensional Evaluation**: Scores answers across four dimensions:
  - Content Relevance (35%)
  - Technical Accuracy (35%)
  - Communication Clarity (15%)
  - STAR Structure (15%)
- **Interactive Web Interface**: User-friendly frontend for practice sessions

### Key Innovation

The system employs a **two-stage transfer learning approach**:
1. **Stage 1**: Pre-training on ASAP-AES essay scoring dataset (12,978 samples)
2. **Stage 2**: Fine-tuning on domain-specific interview Q&A pairs (100 annotated samples)

This approach achieves strong performance with minimal domain-specific data by leveraging transfer learning from a related task.

---

## System Architecture

```
┌─────────────────┐
│   Web Frontend  │
│  (HTML/CSS/JS)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI Backend│
│   (REST API)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│  RAG   │ │   Scoring    │
│Retriever│ │   System    │
└────────┘ └──────┬───────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
    ┌─────────┐      ┌──────────┐
    │  BERT   │      │Heuristic │
    │  Model  │      │  Rules   │
    └─────────┘      └──────────┘
```

### Components

1. **Question Retrieval Engine** (`services/retriever.py`)
   - TF-IDF vectorization of question database
   - Cosine similarity search
   - Filtering by topic, difficulty, question type

2. **Scoring System** (`services/model_eval.py`, `services/eval.py`)
   - **Hybrid Approach**: Combines model predictions with heuristic rules
   - **Model Component**: DistilBERT-based neural network
   - **Heuristic Component**: Keyword matching, STAR structure detection, length analysis

3. **Session Management** (`services/sessions.py`)
   - Tracks user progress
   - Supports free practice and mock interview modes
   - Exports session history

---

## Features

### Core Features

- ✅ **Intelligent Question Selection**: RAG-based retrieval with filtering options
- ✅ **Multi-dimensional Scoring**: Four distinct evaluation dimensions
- ✅ **Hybrid Scoring**: Combines model accuracy with heuristic stability
- ✅ **STAR Structure Detection**: Evaluates Situation-Task-Action-Result format
- ✅ **Progress Tracking**: Session history and performance analytics
- ✅ **Mock Interview Mode**: Fixed-length practice sessions
- ✅ **Export Functionality**: Download session data as JSON

### Scoring Methods

The system supports three scoring modes (configurable per session):

1. **Heuristic Only**: Fast, rule-based scoring (no model required)
2. **Model Only**: Pure BERT-based predictions (requires trained model)
3. **Hybrid** (Default): 70% model + 30% heuristic (recommended for production)

---

## Data Preparation

> **Note**: Due to GitHub file size limitations, training data and model checkpoints are not included in the repository. Follow the steps below to generate/obtain them.

### Step 1: Download ASAP Dataset (for Pre-training)

The ASAP-AES dataset is used for Stage 1 pre-training to learn general text scoring capabilities.

1. **Access Kaggle**:
   - Visit: https://www.kaggle.com/c/asap-aes/data
   - Create a free Kaggle account if needed
   - Accept the competition rules

2. **Download the Dataset**:
   - Download `training_set_rel3.tsv` (~50MB)
   - Save to: `ai_interview_coach/data/training_data/training_set_rel3.tsv`

3. **Process the Data**:
   ```bash
   cd ai_interview_coach/scripts
   bash prepare_asap_data.sh
   ```

   **Expected Output**: `data/training_data/asap_essays.csv` (~9,000 essays from sets 2-6)

   **What the script does**:
   - Filters essay sets 2-6 (appropriate score ranges)
   - Normalizes scores to 0-5 range
   - Extracts essay text and scores
   - Creates CSV format for training

### Step 2: Generate Interview Data for Annotation

The system generates synthetic interview Q&A pairs that need manual annotation.

```bash
cd ai_interview_coach/scripts
python generate_interview_data.py
```

**What this script does**:
- Selects 20 diverse questions from the Kaggle interview database
- Generates 5 answer templates per question with varying quality levels
- Creates 100 Q&A pairs ready for annotation
- Uses fixed random seed (seed=42) for reproducibility

**Output Files**:
- `data/training_data/interview_answers_to_annotate.json` (100 Q&A pairs)
- `data/training_data/annotation_template.txt` (annotation guidelines)

**Answer Quality Levels Generated**:
- **Excellent** (5/5): Comprehensive, STAR-structured, technically accurate
- **Good** (4/5): Solid answer with minor gaps
- **Average** (3/5): Basic understanding, limited detail
- **Below Average** (2/5): Significant issues, vague
- **Poor** (1/5): Incorrect or off-topic

### Step 3: Manual Annotation

**Time Required**: 2-3 hours (can be split among team members)

#### Annotation Process

1. **Use Interactive Tool** (Recommended):
   ```bash
   cd ai_interview_coach/scripts
   python annotation_helper.py
   ```

   Features:
   - Step-by-step display of each Q&A pair
   - Current scores shown for easy adjustment
   - Automatic weighted average calculation
   - Progress saving (can pause and resume)

2. **Or Manual JSON Editing**:
   - Open `data/training_data/interview_answers_to_annotate.json`
   - For each entry, adjust:
     - `overall_score`: Overall quality (1-5)
     - `breakdown.content_relevance`: How well answer addresses question
     - `breakdown.technical_accuracy`: Correctness of technical content
     - `breakdown.communication_clarity`: Expression quality
     - `breakdown.structure_star`: Presence of STAR structure
   - Optionally edit `answer` text to make it more realistic
   - Save as: `data/training_data/interview_answers_annotated.json`

#### Scoring Guidelines

| Score | Description | Example |
|-------|-------------|---------|
| **5.0** | Excellent | Comprehensive, accurate, clear STAR structure, specific examples |
| **4.0** | Good | Solid answer, minor gaps, some structure |
| **3.0** | Average | Basic understanding, limited detail, weak structure |
| **2.0** | Below Average | Significant issues, vague, no structure |
| **1.0** | Poor | Incorrect, off-topic, incomprehensible |

#### Annotation Tips

- **Edit Answer Text**: The generated answers are templates. Improve them with:
  - Specific technical details (algorithms, metrics, tools)
  - Real project examples
  - Quantifiable results
  - Complete STAR structure

- **Team Collaboration**: Since the script uses a fixed seed, all team members generate identical data. You can safely split annotation:
  - Member A: Q01-Q10 (50 samples)
  - Member B: Q11-Q20 (50 samples)
  - Merge JSON files when complete

---

## Model Architecture & Training

### Model Architecture

**Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Why DistilBERT**: 60% smaller and 60% faster than BERT while retaining 97% of performance
- **Parameters**: ~66M parameters
- **Input**: Question + Answer concatenated (max 512 tokens)

**Scoring Heads**: Four independent linear layers
- Content Relevance Head
- Technical Accuracy Head
- Communication Clarity Head
- STAR Structure Head

**Output**: 
- Four dimension scores (0-5 range during training, 1-5 for inference)
- Overall score: Weighted average (35% content, 35% technical, 15% communication, 15% structure)

### Training Strategy: Two-Stage Transfer Learning

#### Stage 1: Pre-training on ASAP Dataset

**Purpose**: Learn general text scoring capabilities from essay scoring task

**Dataset**: ASAP-AES (Automated Student Assessment Prize - Automated Essay Scoring)
- **Size**: ~9,000 essays from sets 2-6
- **Score Range**: Normalized to 0-5
- **Domain**: Academic essays (different from interview answers, but related task)

**Training Configuration**:
```bash
cd ai_interview_coach/models
python train.py \
  --stage 1 \
  --asap_path ../data/training_data/asap_essays.csv \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --device cuda \
  --save_dir ../checkpoints/stage1
```

**Expected Results**:
- Validation MAE: ~0.4-0.6
- Validation Accuracy (±1 point): ~85-90%
- Training Time: 2-3 hours on GPU, 12-24 hours on CPU

**Outputs**:
- `checkpoints/stage1/best_model.pt` (~760MB)
- `checkpoints/stage1/training_history.json`

#### Stage 2: Fine-tuning on Interview Data

**Purpose**: Adapt the model to interview answer scoring domain

**Dataset**: Manually annotated interview Q&A pairs
- **Size**: 100 samples (20 questions × 5 quality levels)
- **Score Range**: 1-5 (converted to 0-5 for training)
- **Domain**: Technical interview answers

**Training Configuration**:
```bash
python train.py \
  --stage 2 \
  --interview_path ../data/training_data/interview_answers_annotated.json \
  --load_checkpoint ../checkpoints/stage1/best_model.pt \
  --batch_size 8 \
  --epochs 20 \
  --learning_rate 5e-6 \
  --freeze_bert \
  --device cuda \
  --save_dir ../checkpoints/stage2
```

**Key Design Decisions**:
- **Lower Learning Rate** (5e-6): Prevents catastrophic forgetting of Stage 1 knowledge
- **Freeze BERT** (`--freeze_bert`): Only fine-tune scoring heads, not the entire encoder
- **More Epochs** (20): Compensate for small dataset size
- **Smaller Batch Size** (8): Better gradient estimates with limited data

**Expected Results**:
- Validation MAE: ~0.3-0.5
- Validation Accuracy (±1 point): ~90-95%
- Training Time: 30-60 minutes on GPU, 2-4 hours on CPU

**Outputs**:
- `checkpoints/stage2/best_model.pt` (~760MB) - **Final model for deployment**
- `checkpoints/stage2/training_history.json`

---

## Training Results

### Model Performance Metrics

Based on evaluation on the test set (15% split from annotated data):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 0.95 | Mean absolute error (target: < 0.5) |
| **RMSE** | 1.09 | Root mean square error (target: < 0.7) |
| **Correlation** | 0.85 | Pearson correlation with ground truth |
| **Accuracy (±0.5)** | 13.3% | Within 0.5 points |
| **Accuracy (±1.0)** | 60.0% | Within 1.0 point |
| **Accuracy (±1.5)** | 86.7% | Within 1.5 points |

**Note**: The MAE of 0.95 is higher than ideal, likely due to:
- Limited training data (100 samples)
- Domain gap between ASAP essays and interview answers
- Small test set size (15 samples)

### Dimension-wise Performance

The model's performance across the four scoring dimensions:

| Dimension | Mean Score | Std Dev | Min | Max |
|-----------|------------|---------|-----|-----|
| **Content Relevance** | 2.52 | 0.78 | 1.30 | 4.07 |
| **Technical Accuracy** | 2.35 | 0.75 | 1.05 | 4.11 |
| **Communication Clarity** | 2.81 | 0.55 | 1.90 | 3.99 |
| **STAR Structure** | 2.79 | 0.50 | 1.97 | 3.92 |

**Insights**:
- Communication and Structure scores show less variance (more consistent)
- Content and Technical scores have wider ranges, indicating the model captures more nuanced differences
- All dimensions show reasonable score distributions across the 1-5 range

### Training History

**Stage 1** (ASAP Pre-training, 3 epochs):
- Final Train Loss: 0.42
- Final Val Loss: 0.49
- Final Val MAE: 0.54
- Final Val RMSE: 0.70
- **Training Curve**: Loss decreased steadily from 0.73 to 0.42

**Stage 2** (Interview Fine-tuning, 22 epochs total):
- Final Train Loss: 1.05
- Final Val Loss: 1.23
- Final Val MAE: 1.01
- Final Val RMSE: 1.11
- **Training Curve**: 
  - Initial jump in loss (epoch 4: 2.74) due to domain shift
  - Gradual decrease to ~1.23 by epoch 22
  - Val MAE improved from 1.46 to 1.01
  - Model converged around epoch 20

**Key Observations**:
- Stage 1 achieved good performance on essay scoring (MAE 0.54)
- Stage 2 loss increased initially (domain shift from essays to interviews), then decreased
- Model converged after ~20 epochs in Stage 2
- Some overfitting observed (train loss 1.05 < val loss 1.23)
- Training curves show stable convergence (see `training_curves.png`)

**Visualization**: Training curves are available in `training_curves.png` showing loss and metric trends across both stages.

### Performance Analysis

**Key Findings**:
- **MAE of 0.95**: Average error of ~1 point on 1-5 scale indicates room for improvement
- **Correlation of 0.85**: Strong correlation suggests the model captures relative quality differences well, even if absolute scores have error
- **Accuracy (±1.0) of 60%**: Model predictions are within 1 point of ground truth for 60% of samples
- **Accuracy (±1.5) of 86.7%**: Most predictions are within 1.5 points, indicating reasonable performance for practical use

**Analysis**:
- **Transfer Learning**: Two-stage training (ASAP pre-training + interview fine-tuning) helps adapt the model to the interview domain
- **Hybrid Approach**: Combining model (70%) with heuristic (30%) provides best balance of accuracy and stability in production
- **Correlation Strength**: 0.85 correlation indicates model captures relative quality differences well, even if absolute scores have error
- **Limited Data Impact**: Small training set (100 samples) and test set (15 samples) limit model performance and evaluation reliability

### Error Analysis

Based on evaluation predictions (see `evaluation_results/predictions.json`):

**Common Error Patterns**:
1. **Under-scoring High-Quality Answers**: Model tends to underestimate excellent answers (4.5-5.0 range)
   - Example: Predicted 3.13 vs Actual 4.90 (error: -1.77)
   - Likely due to limited high-quality examples in training

2. **Better Performance on Low-Quality Answers**: Model more accurately identifies poor answers (1.0-2.0 range)
   - Example: Predicted 1.72 vs Actual 1.20 (error: +0.52)
   - Suggests model learned clear negative signals

3. **Dimension Consistency**: Communication and Structure scores show less variance, indicating more consistent evaluation

**Visualization**: Detailed evaluation plots including predicted vs actual scatter plots, error distributions, and dimension-wise analysis are available in `evaluation_results/evaluation_plots.png`.

### Limitations in Results

1. **Small Test Set**: Only 15 samples in test set limits statistical reliability
2. **Data Imbalance**: Training data may not fully represent all quality levels
3. **Domain Gap**: ASAP essays (longer, academic) vs interview answers (concise, technical) creates transfer challenges
4. **Overfitting**: Train/val loss gap suggests model memorized some training patterns

**Recommendations for Improvement**:
- Collect 200-500 annotated interview samples for better generalization
- Use data augmentation (paraphrasing) to increase effective training size
- Consider ensemble methods combining multiple models
- Fine-tune on domain-specific technical content (Stack Overflow, technical forums)

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone Repository

```bash
git clone https://github.com/yg3002-YifanGE/gw.git
cd gw/ai_interview_coach
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch` - PyTorch for model training
- `transformers` - HuggingFace transformers (DistilBERT)
- `fastapi` - Web framework
- `scikit-learn` - TF-IDF and utilities
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization

### Step 3: Create Directory Structure

```bash
mkdir -p data/training_data data/index checkpoints evaluation_results
```

### Step 4: Prepare Data

Follow the [Data Preparation](#data-preparation) section above to:
1. Download ASAP dataset
2. Generate interview data
3. Annotate interview data

### Step 5: Train Model (Optional)

If you want to train your own model, you have two options:

**Option A: Quick Training Script (Recommended)**
```bash
# Automatically detects device (CPU/GPU/MPS) and sets optimal parameters
./quick_train.sh
```

**Option B: Manual Training**
```bash
# Stage 1: Pre-training
cd models
python train.py --stage 1 \
  --asap_path ../data/training_data/asap_essays.csv \
  --epochs 10 \
  --save_dir ../checkpoints/stage1

# Stage 2: Fine-tuning
python train.py --stage 2 \
  --interview_path ../data/training_data/interview_answers_annotated.json \
  --load_checkpoint ../checkpoints/stage1/best_model.pt \
  --epochs 20 \
  --save_dir ../checkpoints/stage2
```

**Note**: If you don't train, the system will automatically fall back to heuristic-only scoring.

**Helper Scripts Available**:
- `quick_train.sh` - Auto-detect device and run full training pipeline
- `setup_env.sh` - Set up Python virtual environment
- `docker_run.sh` - Run application in Docker container

### Step 6: Build TF-IDF Index

```bash
# Start the application first
uvicorn app.main:app --reload

# In another terminal, build the index
curl -X POST "http://localhost:8000/api/index/build"
```

---

## Usage

### Starting the Application

**Option 1: Direct Start**
```bash
cd ai_interview_coach
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 2: Docker (if available)**
```bash
cd ai_interview_coach
bash docker_run.sh
```

**Access Points**:
- **Web UI**: http://localhost:8000/app/
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Using the Web Interface

1. **Start a Session**:
   - Enter target role (e.g., "Data Scientist")
   - Select filters (topic, question type, difficulty)
   - Choose mode: Free Practice or Mock Interview
   - Click "Start"

2. **Get a Question**:
   - Click "Get Question"
   - Read the question and metadata

3. **Submit an Answer**:
   - Type your answer (use STAR structure)
   - Click "Submit Answer"
   - View feedback with scores and suggestions

4. **View Progress**:
   - Click "View Progress" to see session history
   - Click "View Summary" for aggregated statistics
   - Export session data as JSON

### Using the API

#### Start a Session

```bash
curl -X POST "http://localhost:8000/api/session/start" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "role": "Data Scientist"
    },
    "config": {
      "scoring_method": "hybrid",
      "model_weight": 0.7,
      "filters": {
        "topic": "Machine Learning",
        "difficulty": "medium"
      }
    }
  }'
```

#### Get a Question

```bash
curl "http://localhost:8000/api/session/{session_id}/question"
```

#### Submit an Answer

```bash
curl -X POST "http://localhost:8000/api/session/{session_id}/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "answer_text": "Your answer here..."
  }'
```

#### View Progress

```bash
curl "http://localhost:8000/api/session/{session_id}/progress"
```

---

## API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/index/build` | POST | Build/rebuild TF-IDF index |
| `/api/meta/options` | GET | Get available topics/qtypes/difficulties |
| `/api/session/start` | POST | Start a new session |
| `/api/session/start_mock` | POST | Start mock interview |
| `/api/session/{id}/question` | GET | Get next question |
| `/api/session/{id}/answer` | POST | Submit answer, get feedback |
| `/api/session/{id}/progress` | GET | Get session progress |
| `/api/session/{id}/summary` | GET | Get session summary |
| `/api/session/{id}/export` | GET | Export session as JSON |

### Response Format

**Answer Feedback Response**:
```json
{
  "overall_score": 4.2,
  "breakdown": {
    "content_relevance": 4.3,
    "technical_accuracy": 4.1,
    "communication_clarity": 4.0,
    "structure_star": 4.5
  },
  "strengths": [
    "Strong content relevance to the question",
    "Clear STAR structure present"
  ],
  "improvements": [
    "Add more technical details"
  ],
  "tips": [
    "Reference key concepts from the question directly"
  ],
  "method": "hybrid",
  "model_score": 4.3,
  "heuristic_score": 4.0,
  "blend_ratio": "70% model / 30% heuristic"
}
```

See http://localhost:8000/docs for interactive API documentation.

---

## Limitations & Future Work

### Current Limitations

1. **Limited Training Data**
   - Only 100 annotated interview samples
   - Small test set (15 samples) limits evaluation reliability
   - Model performance (MAE 0.95) could be improved with more data

2. **Domain Gap**
   - ASAP essays (academic, longer) vs. interview answers (concise, technical)
   - Transfer learning helps but doesn't fully bridge the gap
   - Some overfitting observed in Stage 2

3. **Computational Requirements**
   - Model file size: ~760MB (too large for GitHub)
   - Inference time: 1-2 seconds per answer on CPU
   - Requires GPU for efficient training

4. **Scoring Accuracy**
   - MAE of 0.95 means average error of ~1 point on 1-5 scale
   - Heuristic fallback provides stability but lower accuracy
   - Hybrid approach balances both but not perfect

5. **Language Support**
   - Currently English only
   - Tokenizer and model are English-specific

6. **Question Database**
   - Limited to Kaggle interview questions
   - No dynamic question generation
   - Filtering options are fixed

### Future Improvements

1. **Data Collection**
   - Collect 500-1000 real interview Q&A pairs
   - Use active learning to select most informative samples
   - Explore other pre-training datasets (Stack Overflow, technical forums)

2. **Model Improvements**
   - Experiment with larger models (BERT-base, RoBERTa)
   - Add attention visualization for interpretability
   - Implement ensemble methods
   - Fine-tune on domain-specific technical content

3. **Scoring Enhancements**
   - Add knowledge graph for technical accuracy verification
   - Implement semantic similarity with reference answers
   - Add grammar and style checking
   - Support for code snippets in answers

4. **System Features**
   - Real-time feedback during typing
   - Comparison with other users' answers
   - Personalized difficulty adjustment
   - Multi-language support

5. **Deployment**
   - Model quantization for smaller file size
   - API rate limiting and caching
   - User authentication and progress tracking
   - Mobile app version

---

## Project Structure

```
gw/
├── ai_interview_coach/              # Main application
│   ├── app/                         # FastAPI application
│   │   ├── main.py                  # API endpoints
│   │   ├── models.py                # Pydantic models
│   │   └── deps.py                  # Dependencies
│   ├── services/                     # Core services
│   │   ├── eval.py                  # Heuristic scoring
│   │   ├── model_eval.py            # Model & hybrid scoring
│   │   ├── retriever.py             # RAG retrieval
│   │   └── sessions.py              # Session management
│   ├── models/                      # Training code
│   │   ├── answer_scorer.py         # BERT model architecture
│   │   ├── data_loader.py           # Dataset loaders
│   │   ├── train.py                 # Training script
│   │   └── evaluate.py               # Evaluation script
│   ├── scripts/                     # Utility scripts
│   │   ├── generate_interview_data.py  # Data generation
│   │   ├── annotation_helper.py       # Annotation tool
│   │   └── prepare_asap_data.sh        # ASAP data prep
│   ├── web/                         # Frontend
│   │   ├── index.html               # Main UI
│   │   ├── app.js                   # Frontend logic
│   │   └── styles.css               # Styling
│   ├── data/                        # Data directory (not in repo)
│   │   ├── training_data/           # Training datasets
│   │   └── index/                   # TF-IDF index
│   ├── checkpoints/                 # Model checkpoints (not in repo)
│   │   ├── stage1/                 # Pre-training checkpoints
│   │   └── stage2/                 # Fine-tuned checkpoints
│   └── evaluation_results/          # Evaluation outputs
│       ├── evaluation_plots.png     # Visualization charts
│       ├── evaluation_results.json  # Detailed metrics
│       └── predictions.json         # Sample predictions
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Docker configuration
│   └── README.md                    # This file
├── kaggle_data/                     # Interview question database
└── README.md                        # Project overview
```

### Files Not Included in Repository

Due to GitHub file size limitations (100MB per file, repository size limits), the following are **not** included in the repository:

| File/Directory | Size | How to Obtain |
|----------------|------|---------------|
| `data/training_data/asap_essays.csv` | ~9MB | Download from Kaggle (see [Data Preparation](#data-preparation)) |
| `data/training_data/interview_answers_annotated.json` | ~100KB | Generate and annotate (see [Data Preparation](#data-preparation)) |
| `checkpoints/stage1/best_model.pt` | ~760MB | Train Stage 1 (see [Model Training](#model-architecture--training)) |
| `checkpoints/stage2/best_model.pt` | ~760MB | Train Stage 2 (see [Model Training](#model-architecture--training)) |
| `data/index/` | ~10-50MB | Generated automatically by index building API |
| `venv/` | ~500MB+ | Python virtual environment (recreate with `pip install -r requirements.txt`) |

**Total Size if Included**: ~2GB+ (exceeds GitHub repository limits)

**Note**: The following result files **are included** in the repository:
- `training_curves.png` - Training loss and metrics visualization (40KB)
- `evaluation_results/evaluation_plots.png` - Evaluation visualizations (138KB)
- `evaluation_results/evaluation_results.json` - Detailed metrics (1KB)
- `evaluation_results/predictions.json` - Sample predictions with errors (8KB)

**To Reproduce the Complete System**:

1. **Data Preparation** (30-60 minutes):
   ```bash
   # Step 1: Download ASAP dataset from Kaggle
   # Step 2: Generate interview data
   cd ai_interview_coach/scripts
   python generate_interview_data.py
   
   # Step 3: Annotate the data (2-3 hours)
   python annotation_helper.py
   ```

2. **Model Training** (2-3 hours on GPU, 14-28 hours on CPU):
   ```bash
   cd ai_interview_coach/models
   python train.py --stage 1 ...  # Stage 1
   python train.py --stage 2 ...  # Stage 2
   ```

3. **Index Building** (automatic on first API call):
   ```bash
   curl -X POST "http://localhost:8000/api/index/build"
   ```

**Alternative**: If you only want to use the application without training:
- The system will automatically use heuristic-only scoring (no model required)
- All features work except model-based scoring
- You can add model scoring later by training or downloading a pre-trained model

---

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

Run the test suite:
```bash
# Test hybrid scoring
python test_hybrid_scoring.py

# Test model evaluation
cd models
python evaluate.py --checkpoint ../checkpoints/stage2/best_model.pt \
  --data_path ../data/training_data/interview_answers_annotated.json \
  --data_type interview
```

---

## Datasets Used

This project uses the following datasets:

### 1. ASAP-AES Dataset
- **Source**: [Kaggle ASAP-AES Competition](https://www.kaggle.com/c/asap-aes/data)
- **Purpose**: Pre-training data for Stage 1 training
- **Size**: ~9,000 essays from sets 2-6
- **Usage**: General text scoring capability transfer learning

### 2. Data Science Interview Questions
- **Source**: [Data Science Interview Questions by Sandy1811, Kaggle (2021)](https://www.kaggle.com/datasets/sandy1811/data-science-interview-questions)
- **Purpose**: Interview question database for RAG retrieval and training data generation
- **Content**: Machine Learning and Deep Learning interview questions with varying difficulty levels
- **Usage**: Question retrieval system and synthetic answer generation for annotation

---

## Repository

**GitHub**: [https://github.com/yg3002-YifanGE/gw](https://github.com/yg3002-YifanGE/gw)

This is a course project for academic purposes.

---

## License

This project is for educational purposes as part of academic coursework.

---

## Acknowledgments

- **DistilBERT**: HuggingFace Transformers library
- **ASAP-AES Dataset**: [Kaggle ASAP-AES Competition](https://www.kaggle.com/c/asap-aes/data)
- **Data Science Interview Questions**: [Sandy1811, Kaggle (2021)](https://www.kaggle.com/datasets/sandy1811/data-science-interview-questions)
- **FastAPI**: Modern web framework for Python
- **PyTorch**: Deep learning framework

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Status**: ✅ Project Complete - Ready for deployment and further development
