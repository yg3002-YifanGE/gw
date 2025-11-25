"""
Interactive annotation helper tool

Makes manual annotation easier with a simple CLI interface
"""
import json
import os
from pathlib import Path


def display_qa(entry, index, total):
    """Display a Q&A pair nicely formatted"""
    print("\n" + "="*70)
    print(f"Question {index}/{total} - ID: {entry['id']}")
    print("="*70)
    
    # Question
    print(f"\n📌 QUESTION:")
    print(f"   {entry['question']}")
    
    # Metadata
    print(f"\n📊 METADATA:")
    meta = entry.get('metadata', {})
    print(f"   Topic: {meta.get('topic', 'N/A')}")
    print(f"   Difficulty: {meta.get('difficulty', 'N/A')}")
    print(f"   Quality Hint: {meta.get('quality_level', 'N/A')}")
    
    # Answer
    print(f"\n💬 ANSWER:")
    answer_lines = entry['answer'].split('\n')
    for line in answer_lines:
        print(f"   {line}")
    
    # Current scores
    print(f"\n📈 CURRENT SCORES:")
    print(f"   Overall: {entry.get('overall_score', 'N/A')}")
    breakdown = entry.get('breakdown', {})
    print(f"   - Content Relevance:      {breakdown.get('content_relevance', 'N/A')}")
    print(f"   - Technical Accuracy:     {breakdown.get('technical_accuracy', 'N/A')}")
    print(f"   - Communication Clarity:  {breakdown.get('communication_clarity', 'N/A')}")
    print(f"   - STAR Structure:         {breakdown.get('structure_star', 'N/A')}")
    
    if entry.get('notes'):
        print(f"\n📝 NOTES: {entry['notes']}")


def get_float_input(prompt, current_value, min_val=1.0, max_val=5.0):
    """Get validated float input from user"""
    while True:
        try:
            user_input = input(f"{prompt} [{current_value}]: ").strip()
            
            if not user_input:  # Keep current value
                return current_value
            
            value = float(user_input)
            
            if min_val <= value <= max_val:
                return value
            else:
                print(f"   ⚠️  Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("   ⚠️  Please enter a valid number")


def annotate_entry(entry, index, total):
    """Annotate a single entry interactively"""
    display_qa(entry, index, total)
    
    print(f"\n{'='*70}")
    print("ANNOTATION - Enter new scores or press Enter to keep current")
    print("Scoring: 1=Poor, 2=Below Avg, 3=Average, 4=Good, 5=Excellent")
    print("="*70)
    
    # Get scores
    breakdown = entry.get('breakdown', {})
    
    content = get_float_input(
        "Content Relevance (addresses question)",
        breakdown.get('content_relevance', 3.0)
    )
    
    technical = get_float_input(
        "Technical Accuracy (correctness)",
        breakdown.get('technical_accuracy', 3.0)
    )
    
    clarity = get_float_input(
        "Communication Clarity (expression)",
        breakdown.get('communication_clarity', 3.0)
    )
    
    structure = get_float_input(
        "STAR Structure (Situation/Task/Action/Result)",
        breakdown.get('structure_star', 3.0)
    )
    
    # Calculate overall score (weighted average)
    overall = round(
        0.35 * content + 0.35 * technical + 
        0.15 * clarity + 0.15 * structure,
        2
    )
    
    print(f"\n   ✓ Calculated Overall Score: {overall}")
    
    # Optional: Override overall score
    override = input(f"   Override overall score? [Enter to use {overall}]: ").strip()
    if override:
        try:
            overall = float(override)
        except ValueError:
            pass
    
    # Get notes
    print(f"\n   Notes (optional, press Enter to skip):")
    notes = input("   > ").strip()
    
    # Update entry
    entry['overall_score'] = overall
    entry['breakdown'] = {
        'content_relevance': content,
        'technical_accuracy': technical,
        'communication_clarity': clarity,
        'structure_star': structure
    }
    if notes:
        entry['notes'] = notes
    
    print(f"\n   ✓ Annotation saved for {entry['id']}")
    
    return entry


def main():
    """Main annotation loop"""
    print("\n" + "="*70)
    print("AI Interview Coach - Annotation Helper Tool")
    print("="*70)
    
    # Find input file
    data_dir = Path(__file__).parent.parent / 'data' / 'training_data'
    input_file = data_dir / 'interview_answers_to_annotate.json'
    output_file = data_dir / 'interview_answers_annotated.json'
    
    if not input_file.exists():
        print(f"\n❌ Error: Input file not found at {input_file}")
        print("   Please run generate_interview_data.py first")
        return
    
    # Load data
    print(f"\n📂 Loading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} Q&A pairs")
    
    # Check if there's a partial save
    if output_file.exists():
        print(f"\n⚠️  Found existing annotations at {output_file}")
        resume = input("   Resume from where you left off? (y/n): ").strip().lower()
        
        if resume == 'y':
            with open(output_file, 'r', encoding='utf-8') as f:
                annotated_data = json.load(f)
            
            # Merge with existing
            annotated_ids = {entry['id'] for entry in annotated_data}
            data = [entry for entry in data if entry['id'] not in annotated_ids]
            
            print(f"✓ Resuming with {len(data)} remaining entries")
            print(f"✓ Already completed: {len(annotated_data)} entries")
        else:
            annotated_data = []
    else:
        annotated_data = []
    
    # Annotation loop
    total = len(data)
    
    for i, entry in enumerate(data, 1):
        try:
            annotated = annotate_entry(entry, i, total)
            annotated_data.append(annotated)
            
            # Auto-save after each annotation
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotated_data, f, indent=2, ensure_ascii=False)
            
            # Ask if user wants to continue
            if i < total:
                print(f"\n{'─'*70}")
                choice = input(f"Continue to next? (y/n/q to quit): ").strip().lower()
                
                if choice == 'q':
                    print("\n📊 Progress saved. You can resume later.")
                    break
                elif choice == 'n':
                    print("\n📊 Pausing. Run this script again to resume.")
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Progress saved.")
            break
    
    # Final save and summary
    print(f"\n{'='*70}")
    print("ANNOTATION COMPLETE!")
    print("="*70)
    print(f"✓ Total annotated: {len(annotated_data)} Q&A pairs")
    print(f"✓ Saved to: {output_file}")
    
    # Statistics
    if annotated_data:
        avg_score = sum(e['overall_score'] for e in annotated_data) / len(annotated_data)
        print(f"\n📊 Statistics:")
        print(f"   Average Score: {avg_score:.2f}")
        print(f"   Min Score: {min(e['overall_score'] for e in annotated_data):.2f}")
        print(f"   Max Score: {max(e['overall_score'] for e in annotated_data):.2f}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Review your annotations in {output_file.name}")
    print(f"   2. Run training: python models/train.py --stage 2 ...")
    print()


if __name__ == '__main__':
    main()

