"""
Evaluation script for answer scoring model
"""
import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from answer_scorer import BERTAnswerScorer
from data_loader import create_data_loaders


def evaluate_model(model, data_loader, device='cuda'):
    """
    Comprehensive evaluation
    
    Returns detailed metrics and predictions
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_texts = []
    all_breakdowns = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_scores = batch['score'].to(device)
            
            outputs = model(input_ids, attention_mask)
            predicted_scores = outputs['overall_score'].squeeze()
            predicted_breakdown = outputs['breakdown']
            
            all_predictions.extend(predicted_scores.cpu().numpy())
            all_targets.extend(target_scores.cpu().numpy())
            all_breakdowns.extend(predicted_breakdown.cpu().numpy())
            all_texts.extend(batch['text'])
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_breakdowns = np.array(all_breakdowns)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    mse = np.mean((all_predictions - all_targets) ** 2)
    
    # Accuracy within different thresholds
    acc_05 = np.mean(np.abs(all_predictions - all_targets) <= 0.5)
    acc_10 = np.mean(np.abs(all_predictions - all_targets) <= 1.0)
    acc_15 = np.mean(np.abs(all_predictions - all_targets) <= 1.5)
    
    # Correlation
    correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
    
    # Per-dimension metrics
    dimension_names = ['content', 'technical', 'communication', 'structure']
    dimension_stats = {}
    for i, name in enumerate(dimension_names):
        dimension_stats[name] = {
            'mean': float(np.mean(all_breakdowns[:, i])),
            'std': float(np.std(all_breakdowns[:, i])),
            'min': float(np.min(all_breakdowns[:, i])),
            'max': float(np.max(all_breakdowns[:, i]))
        }
    
    results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'correlation': float(correlation),
        'accuracy': {
            '±0.5': float(acc_05),
            '±1.0': float(acc_10),
            '±1.5': float(acc_15)
        },
        'dimension_stats': dimension_stats,
        'num_samples': len(all_predictions)
    }
    
    return results, all_predictions, all_targets, all_breakdowns, all_texts


def plot_results(predictions, targets, save_path='evaluation_plot.png'):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    ax.plot([1, 5], [1, 5], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title('Predicted vs Actual Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    errors = predictions - targets
    ax.hist(errors, bins=50, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (MAE={np.mean(np.abs(errors)):.3f})')
    ax.legend()
    
    # 3. Distribution of predictions vs targets
    ax = axes[1, 0]
    ax.hist(targets, bins=20, alpha=0.5, label='Actual', edgecolor='black')
    ax.hist(predictions, bins=20, alpha=0.5, label='Predicted', edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distributions')
    ax.legend()
    
    # 4. Cumulative error
    ax = axes[1, 1]
    abs_errors = np.abs(errors)
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cumulative)
    ax.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='±0.5')
    ax.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='±1.0')
    ax.axvline(x=1.5, color='r', linestyle='--', alpha=0.5, label='±1.5')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('Cumulative Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def compare_with_baseline(model, data_loader, device='cuda'):
    """
    Compare model predictions with heuristic baseline
    """
    from services.eval import heuristic_feedback
    
    model.eval()
    model_predictions = []
    baseline_predictions = []
    targets = []
    
    # This would require access to the actual text for heuristic scoring
    # Simplified version here
    print("Comparing with heuristic baseline...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Comparing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_scores = batch['score'].to(device)
            
            # Model prediction
            outputs = model(input_ids, attention_mask)
            predicted_scores = outputs['overall_score'].squeeze()
            
            model_predictions.extend(predicted_scores.cpu().numpy())
            targets.extend(target_scores.cpu().numpy())
    
    model_predictions = np.array(model_predictions)
    targets = np.array(targets)
    
    model_mae = np.mean(np.abs(model_predictions - targets))
    
    results = {
        'model_mae': float(model_mae),
        'improvement': 'N/A (baseline comparison not fully implemented)'
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate answer scoring model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to evaluation data (CSV or JSON)')
    parser.add_argument('--data_type', type=str, choices=['asap', 'interview'],
                       default='interview', help='Type of evaluation data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}")
    
    # Load model
    model = BERTAnswerScorer()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    print(f"Loading data from {args.data_path}")
    
    # Create data loader
    if args.data_type == 'asap':
        from data_loader import create_data_loaders
        loaders = create_data_loaders(
            asap_path=args.data_path,
            batch_size=args.batch_size
        )
        test_loader = loaders['test']
    else:
        from data_loader import create_data_loaders
        loaders = create_data_loaders(
            interview_path=args.data_path,
            batch_size=args.batch_size
        )
        test_loader = loaders['test']
    
    # Evaluate
    print("\nEvaluating model...")
    results, predictions, targets, breakdowns, texts = evaluate_model(
        model, test_loader, args.device
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Number of samples: {results['num_samples']}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    print(f"\nAccuracy:")
    print(f"  Within ±0.5: {results['accuracy']['±0.5']:.2%}")
    print(f"  Within ±1.0: {results['accuracy']['±1.0']:.2%}")
    print(f"  Within ±1.5: {results['accuracy']['±1.5']:.2%}")
    
    print(f"\nDimension Statistics:")
    for dim, stats in results['dimension_stats'].items():
        print(f"  {dim.capitalize()}:")
        print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Create plots
    plot_path = os.path.join(args.output_dir, 'evaluation_plots.png')
    plot_results(predictions, targets, plot_path)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.json')
    predictions_data = []
    for i in range(min(100, len(predictions))):  # Save first 100
        predictions_data.append({
            'text_snippet': texts[i],
            'predicted_score': float(predictions[i]),
            'actual_score': float(targets[i]),
            'error': float(predictions[i] - targets[i]),
            'breakdown': {
                'content': float(breakdowns[i][0]),
                'technical': float(breakdowns[i][1]),
                'communication': float(breakdowns[i][2]),
                'structure': float(breakdowns[i][3])
            }
        })
    
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Saved sample predictions to {predictions_path}")
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == '__main__':
    main()

