"""
Data loading utilities for ASAP-AES and interview datasets
"""
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class ASAPDataset(Dataset):
    """
    ASAP-AES Dataset Loader
    
    The ASAP dataset contains student essays with scores.
    We'll use sets 2-6 which have score ranges of 0-3 or 0-4.
    """
    
    def __init__(self, csv_path, tokenizer, max_length=512, score_range=(0, 4)):
        """
        Args:
            csv_path: Path to ASAP CSV file
            tokenizer: BERT tokenizer
            max_length: Max sequence length
            score_range: Original score range (will be normalized to 1-5)
        """
        self.df = pd.read_csv(csv_path, encoding='latin-1')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_range = score_range
        
        # Filter to specific essay sets (2-6 have reasonable score ranges)
        if 'essay_set' in self.df.columns:
            self.df = self.df[self.df['essay_set'].isin([2, 3, 4, 5, 6])]
        
        # Clean data
        self.df = self.df.dropna(subset=['essay', 'domain1_score'])
        self.df = self.df.reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} essays from ASAP dataset")
        
    def __len__(self):
        return len(self.df)
    
    def _normalize_score(self, score, original_range):
        """
        Normalize score from original range to 0-5
        
        This allows better utilization of all score ranges:
        - Set 2 (1-6): maps to 0-5
        - Set 3-4 (0-3): maps to 0-5  
        - Set 5-6 (0-4): maps to 0-5
        
        Final output can be converted to 1-6 (+1) or 1-5 (scale) if needed.
        """
        min_score, max_score = original_range
        # Normalize to 0-1
        normalized = (score - min_score) / (max_score - min_score)
        # Scale to 0-5
        return 5.0 * normalized
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get essay text (treat as "answer")
        essay = str(row['essay'])
        
        # Get score - domain1_score is the main score
        score = float(row['domain1_score'])
        
        # Determine score range for this essay set
        essay_set = row.get('essay_set', 2)
        if essay_set == 2:
            original_range = (1, 6)  # Will map to 0-5
        elif essay_set in [3, 4]:
            original_range = (0, 3)  # Will map to 0-5
        else:  # 5, 6
            original_range = (0, 4)  # Will map to 0-5
        
        # Normalize to 0-5 (better utilization of score ranges)
        normalized_score = self._normalize_score(score, original_range)
        
        # For ASAP, we don't have separate questions, so we'll create a generic prompt
        prompt = row.get('essay_prompt', 'Write an essay on the given topic.')
        text = f"Prompt: {prompt} Response: {essay}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(normalized_score, dtype=torch.float),
            'text': essay[:200]  # Store snippet for debugging
        }


class InterviewDataset(Dataset):
    """
    Custom Interview Answer Dataset
    
    Expected JSON format:
    [
        {
            "question": "What is backpropagation?",
            "answer": "Backpropagation is...",
            "overall_score": 4.2,
            "breakdown": {
                "content_relevance": 4.5,
                "technical_accuracy": 4.0,
                "communication_clarity": 4.0,
                "structure_star": 4.3
            }
        },
        ...
    ]
    """
    
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} interview Q&As")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        answer = item['answer']
        text = f"Question: {question} Answer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get scores (assume they are in 0-5 range, or convert from 1-5 if needed)
        overall_score = float(item.get('overall_score', 2.5))  # Default to middle of 0-5
        breakdown = item.get('breakdown', {})
        
        # If scores are in 1-5 range, convert to 0-5
        if overall_score > 5.0:
            # Likely in 1-5 range, convert to 0-5
            overall_score = (overall_score - 1.0) * 5.0 / 4.0
        
        breakdown_values = [
            float(breakdown.get('content_relevance', overall_score)),
            float(breakdown.get('technical_accuracy', overall_score)),
            float(breakdown.get('communication_clarity', overall_score)),
            float(breakdown.get('structure_star', overall_score))
        ]
        
        # Convert to 0-5 if needed
        breakdown_values = [
            (v - 1.0) * 5.0 / 4.0 if v > 5.0 else v 
            for v in breakdown_values
        ]
        
        breakdown_tensor = torch.tensor(breakdown_values, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(overall_score, dtype=torch.float),
            'breakdown': breakdown_tensor,
            'text': answer[:200]
        }


def create_data_loaders(asap_path=None, interview_path=None, 
                        batch_size=16, val_split=0.15, test_split=0.15):
    """
    Create train/val/test data loaders
    
    Args:
        asap_path: Path to ASAP CSV
        interview_path: Path to interview JSON
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        dict with train/val/test DataLoaders
    """
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    
    datasets = []
    
    # Load ASAP if provided
    if asap_path and os.path.exists(asap_path):
        asap_dataset = ASAPDataset(asap_path, tokenizer)
        datasets.append(asap_dataset)
    
    # Load interview data if provided
    if interview_path and os.path.exists(interview_path):
        interview_dataset = InterviewDataset(interview_path, tokenizer)
        datasets.append(interview_dataset)
    
    if not datasets:
        raise ValueError("No valid dataset provided")
    
    # Combine datasets
    if len(datasets) > 1:
        combined = torch.utils.data.ConcatDataset(datasets)
    else:
        combined = datasets[0]
    
    # Split into train/val/test
    total_size = len(combined)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        combined, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

