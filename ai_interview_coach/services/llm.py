import os
from typing import Dict, Any, List


def llm_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_feedback_with_llm(prompt: str) -> Dict[str, Any]:
    """Optional OpenAI-based feedback generator.
    Returns a structured dict compatible with Feedback.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an interview coach. Return strict JSON with keys: overall_score, breakdown{content_relevance,technical_accuracy,communication_clarity,structure_star}, strengths, improvements, tips."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        import json
        content = resp.choices[0].message.content
        data = json.loads(content)
        # basic sanitation
        data["overall_score"] = float(data.get("overall_score", 0))
        return data
    except Exception:
        # Fall back to heuristic if anything goes wrong
        return {}

