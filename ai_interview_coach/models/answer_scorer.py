"""
BERT-based Answer Scoring Model
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BERTAnswerScorer(nn.Module):
    """
    BERT-based model for scoring interview answers (1-5 scale)
    
    Architecture:
    - BERT encoder (distilbert-base-uncased)
    - Dropout layer
    - Linear projection to 4 dimensions (content, technical, communication, structure)
    - Overall score as weighted average
    """
    
    def __init__(self, model_name='distilbert-base-uncased', dropout=0.3, freeze_bert=False):
        super().__init__()
        
        # Load pre-trained BERT
        import sys
        print(f"    - Loading DistilBERT model '{model_name}' (this may take a moment if downloading)...", flush=True)
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(model_name)
        print("    - DistilBERT model loaded", flush=True)
        print(f"    - Loading tokenizer '{model_name}'...", flush=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        print("    - Tokenizer loaded", flush=True)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Dimension heads
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Four scoring dimensions
        self.content_head = nn.Linear(hidden_size, 1)
        self.technical_head = nn.Linear(hidden_size, 1)
        self.communication_head = nn.Linear(hidden_size, 1)
        self.structure_head = nn.Linear(hidden_size, 1)
        
        # Weights for overall score (can be learned or fixed)
        self.dimension_weights = nn.Parameter(
            torch.tensor([0.35, 0.35, 0.15, 0.15]), requires_grad=False
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            dict with:
                - overall_score: [batch_size, 1] overall score (1-5)
                - breakdown: [batch_size, 4] dimension scores
        """
        # Get BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Get dimension scores
        content_score = self.content_head(pooled_output)
        technical_score = self.technical_head(pooled_output)
        communication_score = self.communication_head(pooled_output)
        structure_score = self.structure_head(pooled_output)
        
        # Stack dimension scores [batch_size, 4]
        breakdown = torch.cat([
            content_score, technical_score, 
            communication_score, structure_score
        ], dim=1)
        
        # Scale to 0-5 range using sigmoid: 5 * sigmoid(x)
        # This matches the training data normalization (0-5)
        # Can be converted to 1-6 (+1) or 1-5 (scale) in final output if needed
        breakdown = 5.0 * torch.sigmoid(breakdown)
        
        # Weighted average for overall score
        overall_score = torch.sum(breakdown * self.dimension_weights, dim=1, keepdim=True)
        
        return {
            'overall_score': overall_score,
            'breakdown': breakdown
        }
    
    def predict(self, question, answer, max_length=512, output_range='0-5'):
        """
        Predict score for a single question-answer pair
        
        Args:
            question: Question text
            answer: Answer text
            max_length: Max sequence length
            output_range: '0-5' (default), '1-6' (+1), or '1-5' (scale)
            
        Returns:
            dict with overall_score and breakdown
        """
        self.eval()
        
        # Combine question and answer
        text = f"Question: {question} Answer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict (model outputs 0-5)
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        # Convert to numpy
        overall_score = outputs['overall_score'].cpu().item()
        breakdown = outputs['breakdown'].cpu().numpy()[0]
        
        # Convert to desired output range
        if output_range == '1-6':
            # Simple shift: 0-5 -> 1-6
            overall_score = overall_score + 1.0
            breakdown = breakdown + 1.0
        elif output_range == '1-5':
            # Scale: 0-5 -> 1-5
            overall_score = overall_score * 4.0 / 5.0 + 1.0
            breakdown = breakdown * 4.0 / 5.0 + 1.0
        # else: keep 0-5 (default)
        
        return {
            'overall_score': round(overall_score, 2),
            'breakdown': {
                'content_relevance': round(float(breakdown[0]), 2),
                'technical_accuracy': round(float(breakdown[1]), 2),
                'communication_clarity': round(float(breakdown[2]), 2),
                'structure_star': round(float(breakdown[3]), 2)
            },
            'raw_score_0_5': round(outputs['overall_score'].cpu().item(), 2)  # Keep original for reference
        }

