"""
Generate synthetic interview Q&A pairs for manual annotation

This script:
1. Reads questions from your kaggle_data
2. Generates 3-5 answers per question with varying quality
3. Outputs JSON file ready for manual annotation
"""
import os
import sys
import json
import random
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from services.retriever import TfIdfRetriever
from app.deps import settings

# Fix data_dir path if running from scripts directory
# settings.data_dir is relative to ai_interview_coach/, but kaggle_data is in gw/
if not Path(settings.data_dir).exists():
    # Try alternative path: go up two levels to gw/kaggle_data
    alt_data_dir = project_root.parent / 'kaggle_data'
    if alt_data_dir.exists():
        settings.data_dir = str(alt_data_dir)
        print(f"Using alternative data directory: {settings.data_dir}")


def generate_diverse_answers(question, num_answers=5):
    """
    Generate answer templates of varying quality
    
    Returns list of dicts with answer text and suggested quality level
    """
    answers = []
    
    # Answer 1: Excellent (5/5) - Comprehensive with STAR
    answers.append({
        "answer": f"[EXCELLENT] Regarding '{question}': In my previous role (Situation), "
                  f"I was tasked with implementing this concept (Task). "
                  f"I approached it by... (Action). As a result, we achieved... (Result). "
                  f"This demonstrates strong understanding of the fundamentals and practical application.",
        "suggested_score": 4.5,
        "quality": "excellent"
    })
    
    # Answer 2: Good (4/5) - Solid answer, less structured
    answers.append({
        "answer": f"[GOOD] '{question}' is an important concept. "
                  f"It involves several key components including... "
                  f"From my experience, I've found that... "
                  f"The main advantages are... However, there are some trade-offs to consider.",
        "suggested_score": 4.0,
        "quality": "good"
    })
    
    # Answer 3: Average (3/5) - Basic understanding
    answers.append({
        "answer": f"[AVERAGE] About '{question}', I know it's related to... "
                  f"Basically, you use it when... I think the main idea is that... "
                  f"I've read about this before and understand the general concept.",
        "suggested_score": 3.0,
        "quality": "average"
    })
    
    # Answer 4: Below Average (2/5) - Incomplete or vague
    answers.append({
        "answer": f"[BELOW_AVERAGE] '{question}'? I'm not entirely sure, but I think... "
                  f"Maybe it has something to do with... "
                  f"I haven't worked with this directly, but I've heard of it.",
        "suggested_score": 2.0,
        "quality": "below_average"
    })
    
    # Answer 5: Poor (1/5) - Incorrect or off-topic
    answers.append({
        "answer": f"[POOR] I don't really know about '{question}'. "
                  f"Could you explain what that is? I haven't encountered this before.",
        "suggested_score": 1.5,
        "quality": "poor"
    })
    
    return answers


def main():
    """Generate interview dataset for annotation"""
    
    print("="*60)
    print("Interview Answer Generation for Manual Annotation")
    print("="*60)
    
    # Initialize retriever
    print("\nLoading question database...")
    retriever = TfIdfRetriever(settings.data_dir, settings.index_dir)
    retriever._ensure_loaded()
    
    # Get diverse questions
    print(f"Total questions available: {len(retriever.meta)}")
    
    # Sample questions across different topics and difficulties
    questions_to_annotate = []
    
    # Strategy: Get 20 questions across different categories
    filters_list = [
        {'topic': 'Machine Learning', 'difficulty': 'easy'},
        {'topic': 'Machine Learning', 'difficulty': 'medium'},
        {'topic': 'Machine Learning', 'difficulty': 'hard'},
        {'topic': 'Deep Learning', 'difficulty': 'easy'},
        {'topic': 'Deep Learning', 'difficulty': 'medium'},
        {'topic': 'Deep Learning', 'difficulty': 'hard'},
    ]
    
    questions_per_category = 4
    
    for filters in filters_list:
        # Find matching questions
        matching = [
            m for m in retriever.meta
            if (filters.get('topic') is None or m.get('topic') == filters['topic'])
            and (filters.get('difficulty') is None or m.get('difficulty') == filters['difficulty'])
        ]
        
        if matching:
            sampled = random.sample(matching, min(questions_per_category, len(matching)))
            questions_to_annotate.extend(sampled)
    
    # Ensure we have at least 20 questions
    if len(questions_to_annotate) < 20:
        remaining = 20 - len(questions_to_annotate)
        additional = random.sample(retriever.meta, remaining)
        questions_to_annotate.extend(additional)
    
    # Deduplicate
    questions_to_annotate = list({q['doc_id']: q for q in questions_to_annotate}.values())
    questions_to_annotate = questions_to_annotate[:20]  # Limit to 20 questions
    
    print(f"\nSelected {len(questions_to_annotate)} questions for annotation")
    
    # Generate dataset
    dataset = []
    
    for i, q_meta in enumerate(questions_to_annotate, 1):
        question_text = q_meta['text']
        
        print(f"\nQuestion {i}/{len(questions_to_annotate)}: {question_text[:60]}...")
        
        # Generate 5 answers of varying quality
        answer_variants = generate_diverse_answers(question_text)
        
        for j, answer_variant in enumerate(answer_variants, 1):
            entry = {
                "id": f"Q{i:02d}_A{j}",
                "question": question_text,
                "answer": answer_variant['answer'],
                "metadata": {
                    "topic": q_meta.get('topic', 'Unknown'),
                    "qtype": q_meta.get('qtype', 'Unknown'),
                    "difficulty": q_meta.get('difficulty', 'Unknown'),
                    "quality_level": answer_variant['quality'],
                    "doc_id": q_meta.get('doc_id', 'Unknown')
                },
                # These fields are to be manually annotated
                "overall_score": answer_variant['suggested_score'],  # Pre-filled suggestion
                "breakdown": {
                    "content_relevance": answer_variant['suggested_score'],
                    "technical_accuracy": answer_variant['suggested_score'],
                    "communication_clarity": answer_variant['suggested_score'],
                    "structure_star": answer_variant['suggested_score']
                },
                "notes": ""  # For annotator comments
            }
            
            dataset.append(entry)
    
    # Save to file
    output_dir = Path(__file__).parent.parent / 'data' / 'training_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'interview_answers_to_annotate.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Generated {len(dataset)} Q&A pairs")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print instructions
    print("\n" + "="*60)
    print("NEXT STEPS - Manual Annotation")
    print("="*60)
    print("""
1. Open the generated JSON file in a text editor or spreadsheet
   
2. For each entry, review the answer and adjust scores (1-5 scale):
   - overall_score: Overall quality of the answer
   - breakdown:
     * content_relevance: How well it addresses the question
     * technical_accuracy: Correctness of technical content
     * communication_clarity: How clearly it's expressed
     * structure_star: Presence of STAR structure (Situation, Task, Action, Result)
   
3. Suggested scoring guidelines:
   - 5.0: Excellent - Comprehensive, accurate, well-structured
   - 4.0: Good - Solid answer with minor gaps
   - 3.0: Average - Basic understanding, some issues
   - 2.0: Below Average - Significant gaps or errors
   - 1.0: Poor - Incorrect or off-topic
   
4. Add any notes in the "notes" field
   
5. Save the file as 'interview_answers_annotated.json' in the same directory
   
6. The [QUALITY_LEVEL] tags in answers are suggestions - adjust as needed!
   
NOTE: You can also edit the answers to make them more realistic if desired.
    """)
    
    print("\nTIP: You can split the annotation work among team members.")
    print("Each person can annotate a subset of the questions.\n")
    
    # Create a simplified annotation template
    template_file = output_dir / 'annotation_template.txt'
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write("ANNOTATION TEMPLATE\n")
        f.write("="*60 + "\n\n")
        f.write("For each Q&A pair, rate on a scale of 1-5:\n\n")
        f.write("1. CONTENT RELEVANCE (Does it address the question?)\n")
        f.write("   1 = Off-topic, 3 = Partially addresses, 5 = Fully addresses\n\n")
        f.write("2. TECHNICAL ACCURACY (Is the information correct?)\n")
        f.write("   1 = Incorrect, 3 = Mostly correct, 5 = Completely accurate\n\n")
        f.write("3. COMMUNICATION CLARITY (Is it well-expressed?)\n")
        f.write("   1 = Confusing, 3 = Understandable, 5 = Very clear\n\n")
        f.write("4. STRUCTURE (STAR: Situation, Task, Action, Result)\n")
        f.write("   1 = No structure, 3 = Some structure, 5 = Clear STAR format\n\n")
        f.write("5. OVERALL SCORE (Weighted average of above)\n")
        f.write("   Consider: 35% content, 35% technical, 15% clarity, 15% structure\n\n")
    
    print(f"✓ Also saved annotation guidelines to: {template_file}\n")


if __name__ == '__main__':
    main()

