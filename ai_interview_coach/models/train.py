"""
Training script for answer scoring model
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from answer_scorer import BERTAnswerScorer
from data_loader import create_data_loaders


class Trainer:
    """
    Trainer for BERT Answer Scorer
    
    Supports two-stage training:
    - Stage 1: Pre-train on ASAP dataset
    - Stage 2: Fine-tune on interview dataset
    """
    
    def __init__(self, model, device='cuda', learning_rate=2e-5, weight_decay=0.01):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': []
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_scores = batch['score'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            predicted_scores = outputs['overall_score'].squeeze()
            
            # Calculate loss
            loss = self.mse_loss(predicted_scores, target_scores)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_scores = batch['score'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predicted_scores = outputs['overall_score'].squeeze()
                
                loss = self.mse_loss(predicted_scores, target_scores)
                total_loss += loss.item()
                
                all_predictions.extend(predicted_scores.cpu().numpy())
                all_targets.extend(target_scores.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        avg_loss = total_loss / len(val_loader)
        
        # Accuracy within ±1 point
        accuracy = np.mean(np.abs(all_predictions - all_targets) <= 1.0)
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader, epochs=10, save_dir='./checkpoints'):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_mae = float('inf')
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f}")
            print(f"Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"Val Accuracy (±1): {val_metrics['accuracy']:.2%}")
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pt'),
                    epoch, val_metrics
                )
                print(f"✓ Saved best model (MAE: {best_val_mae:.4f})")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'),
                    epoch, val_metrics
                )
            
            scheduler.step()
        
        # Save final model
        self.save_checkpoint(
            os.path.join(save_dir, 'final_model.pt'),
            epochs, val_metrics
        )
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best Val MAE: {best_val_mae:.4f}")
        print(f"{'='*50}")
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main():
    parser = argparse.ArgumentParser(description='Train answer scoring model')
    
    # Data arguments
    parser.add_argument('--asap_path', type=str, help='Path to ASAP CSV file')
    parser.add_argument('--interview_path', type=str, help='Path to interview JSON file')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='Training stage: 1=ASAP only, 2=Fine-tune on interview')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--freeze_bert', action='store_true',
                       help='Freeze BERT weights (for stage 2)')
    parser.add_argument('--load_checkpoint', type=str, help='Load from checkpoint')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    print(f"Training Stage {args.stage}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    
    # Create data loaders
    print("\nLoading data...")
    data_loaders = create_data_loaders(
        asap_path=args.asap_path if args.stage == 1 else None,
        interview_path=args.interview_path if args.stage == 2 else None,
        batch_size=args.batch_size
    )
    
    # Create model
    print("\nInitializing model...")
    model = BERTAnswerScorer(
        model_name=args.model_name,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        epoch, metrics = trainer.load_checkpoint(args.load_checkpoint)
        print(f"Loaded from epoch {epoch}, metrics: {metrics}")
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    test_metrics = trainer.evaluate(data_loaders['test'])
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test Accuracy (±1): {test_metrics['accuracy']:.2%}")
    
    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == '__main__':
    main()

