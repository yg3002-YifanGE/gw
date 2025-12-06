#!/bin/bash
# Quick training script - Auto-detect device and start training

echo "=========================================="
echo "🚀 AI Interview Coach - Model Training Script"
echo "=========================================="
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found"
    exit 1
fi

# Check if torch is installed
echo "📦 Checking dependencies..."
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: PyTorch not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Detect device
echo "🔍 Detecting compute device..."
DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")

echo "✅ Detected device: $DEVICE"
echo ""

# Check data files
echo "📁 Checking data files..."
if [ ! -f "data/training_data/asap_essays.csv" ]; then
    echo "❌ Error: ASAP data file not found"
    echo "   Path: data/training_data/asap_essays.csv"
    exit 1
fi

if [ ! -f "data/training_data/interview_answers_annotated.json" ]; then
    echo "❌ Error: Annotated interview data not found"
    echo "   Path: data/training_data/interview_answers_annotated.json"
    exit 1
fi

echo "✅ Data files ready"
echo ""

# Create checkpoints directory
mkdir -p checkpoints/stage1 checkpoints/stage2

# Set batch size based on device
if [ "$DEVICE" = "cpu" ]; then
    BATCH_SIZE_1=8
    BATCH_SIZE_2=4
    EPOCHS_1=5
    EPOCHS_2=10
    echo "⚠️  Using CPU, will use smaller batch size to save memory"
else
    BATCH_SIZE_1=16
    BATCH_SIZE_2=8
    EPOCHS_1=10
    EPOCHS_2=20
fi

echo "=========================================="
echo "📋 Training Configuration"
echo "=========================================="
echo "Device: $DEVICE"
echo "Stage 1 batch size: $BATCH_SIZE_1"
echo "Stage 1 epochs: $EPOCHS_1"
echo "Stage 2 batch size: $BATCH_SIZE_2"
echo "Stage 2 epochs: $EPOCHS_2"
echo ""

# Ask to continue
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Stage 1
echo ""
echo "=========================================="
echo "🎯 Stage 1: ASAP Pre-training"
echo "=========================================="
echo ""

cd models
python3 train.py \
  --stage 1 \
  --asap_path ../data/training_data/asap_essays.csv \
  --batch_size $BATCH_SIZE_1 \
  --epochs $EPOCHS_1 \
  --device $DEVICE \
  --save_dir ../checkpoints/stage1

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 training failed"
    exit 1
fi

# Stage 2
echo ""
echo "=========================================="
echo "🎯 Stage 2: Interview Data Fine-tuning"
echo "=========================================="
echo ""

python3 train.py \
  --stage 2 \
  --interview_path ../data/training_data/interview_answers_annotated.json \
  --load_checkpoint ../checkpoints/stage1/best_model.pt \
  --batch_size $BATCH_SIZE_2 \
  --epochs $EPOCHS_2 \
  --learning_rate 5e-6 \
  --freeze_bert \
  --device $DEVICE \
  --save_dir ../checkpoints/stage2

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 Training complete!"
echo "=========================================="
echo ""
echo "✅ Best model saved to:"
echo "   checkpoints/stage2/best_model.pt"
echo ""
echo "💡 Next step: Run evaluation script to see detailed metrics"
echo "   cd models && python3 evaluate.py ..."
echo ""
