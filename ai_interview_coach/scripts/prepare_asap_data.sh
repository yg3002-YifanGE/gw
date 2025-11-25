#!/bin/bash
# Script to download and prepare ASAP-AES dataset

echo "============================================================"
echo "ASAP-AES Dataset Preparation"
echo "============================================================"

# Create data directory
mkdir -p ../data/training_data

echo ""
echo "Step 1: Download ASAP Dataset"
echo "------------------------------------------------------------"
echo "Please download the ASAP-AES dataset from Kaggle:"
echo ""
echo "  URL: https://www.kaggle.com/c/asap-aes/data"
echo ""
echo "You will need to:"
echo "  1. Sign in to Kaggle (or create a free account)"
echo "  2. Download 'training_set_rel3.tsv'"
echo "  3. Save it to: ../data/training_data/"
echo ""

read -p "Have you downloaded the file? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Please download the dataset first and then run this script again."
    exit 1
fi

# Check if file exists
ASAP_FILE="../data/training_data/training_set_rel3.tsv"

if [ ! -f "$ASAP_FILE" ]; then
    echo "Error: File not found at $ASAP_FILE"
    echo "Please ensure you've downloaded 'training_set_rel3.tsv' to the correct location."
    exit 1
fi

echo ""
echo "Step 2: Convert TSV to CSV"
echo "------------------------------------------------------------"

# Convert TSV to CSV (Python script)
python3 << EOF
import pandas as pd
import os

print("Loading ASAP dataset...")
df = pd.read_csv('$ASAP_FILE', sep='\t', encoding='latin-1')

print(f"Loaded {len(df)} essays")
print(f"Columns: {df.columns.tolist()}")

# Filter to essay sets 2-6 (reasonable score ranges)
df_filtered = df[df['essay_set'].isin([2, 3, 4, 5, 6])]
print(f"Filtered to {len(df_filtered)} essays (sets 2-6)")

# Save as CSV
output_file = '../data/training_data/asap_essays.csv'
df_filtered.to_csv(output_file, index=False)

print(f"Saved to: {output_file}")

# Print statistics
print("\nDataset Statistics:")
print(df_filtered.groupby('essay_set').size())

print("\nScore ranges by essay set:")
for essay_set in [2, 3, 4, 5, 6]:
    subset = df_filtered[df_filtered['essay_set'] == essay_set]
    if len(subset) > 0:
        print(f"  Set {essay_set}: {subset['domain1_score'].min():.0f} - {subset['domain1_score'].max():.0f}")

print("\nPreparation complete!")
EOF

echo ""
echo "============================================================"
echo "✓ ASAP Dataset Ready!"
echo "============================================================"
echo ""
echo "File location: ../data/training_data/asap_essays.csv"
echo ""
echo "Next steps:"
echo "  1. Generate interview data: python scripts/generate_interview_data.py"
echo "  2. Manually annotate the interview data"
echo "  3. Run stage 1 training: python models/train.py --stage 1 --asap_path ..."
echo "  4. Run stage 2 fine-tuning: python models/train.py --stage 2 --interview_path ..."
echo ""

