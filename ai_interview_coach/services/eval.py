from __future__ import annotations

from typing import Dict, Any, List
import re


STAR_KEYS = ["situation", "task", "action", "result"]


def _keyword_score(answer: str, keywords: List[str]) -> float:
    if not answer.strip():
        return 0.0
    a = answer.lower()
    hit = 0
    for kw in keywords:
        if kw and kw.lower() in a:
            hit += 1
    if not keywords:
        return 0.5
    return min(1.0, hit / max(3, len(keywords)))


def _star_structure_score(answer: str) -> float:
    a = answer.lower()
    flags = 0
    for key in STAR_KEYS:
        if re.search(rf"\b{key}\b", a):
            flags += 1
    return flags / 4.0


def heuristic_feedback(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Collect candidate keywords from top retrieved question text
    keywords: List[str] = []
    for r in retrieved[:3]:
        q = r.get("text", "")
        # crude keyword split
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-\+_]{2,}", q):
            if token.lower() not in {"what", "why", "how", "when", "which", "explain"}:
                keywords.append(token)
    keywords = list(dict.fromkeys(keywords))[:12]

    content_rel = _keyword_score(answer, keywords)  # 0..1
    tech_acc = content_rel * 0.9 + 0.1  # naive proxy
    comm = 0.6 if len(answer.split()) >= 40 else 0.4
    star = _star_structure_score(answer)

    # Scale to 1..5
    to5 = lambda x: round(1 + 4 * max(0.0, min(1.0, x)), 2)
    breakdown = {
        "content_relevance": to5(content_rel),
        "technical_accuracy": to5(tech_acc),
        "communication_clarity": to5(comm),
        "structure_star": to5(star),
    }
    overall = round(
        0.35 * breakdown["content_relevance"]
        + 0.35 * breakdown["technical_accuracy"]
        + 0.15 * breakdown["communication_clarity"]
        + 0.15 * breakdown["structure_star"],
        2,
    )

    strengths: List[str] = []
    improvements: List[str] = []
    tips: List[str] = []
    if breakdown["structure_star"] >= 3.5:
        strengths.append("Clear STAR structure present")
    else:
        improvements.append("Use explicit STAR structure: Situation, Task, Action, Result.")
        tips.append("Start with context, then your objective, concrete actions, and measurable outcome.")
    if breakdown["content_relevance"] < 3:
        improvements.append("Increase topical relevance: reference key terms from the question.")
    if breakdown["technical_accuracy"] < 3:
        tips.append("Add definitions, formulas, or short examples to demonstrate correctness.")
    if breakdown["communication_clarity"] < 3:
        tips.append("Aim for concise paragraphs and avoid rambling.")

    return {
        "overall_score": overall,
        "breakdown": breakdown,
        "strengths": strengths,
        "improvements": improvements,
        "tips": tips,
        "evidence": {"keywords": keywords},
    }

