from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .models import (
    StartSessionRequest,
    StartSessionResponse,
    QuestionResponse,
    AnswerRequest,
    AnswerResponse,
    ProgressResponse,
    SummaryResponse,
)
from .deps import settings
from services.sessions import SessionStore
from services.retriever import TfIdfRetriever
from services.eval import heuristic_feedback
from services.llm import llm_enabled, generate_feedback_with_llm
from services.model_eval import hybrid_feedback, model_feedback


app = FastAPI(
    title="AI Interview Coach",
    version="0.2.0",
    description="Interactive interview practice API with RAG retrieval, rubric feedback, filters and mock interviews.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()
retriever = TfIdfRetriever(settings.data_dir, settings.index_dir)

# Serve simple front-end UI
app.mount("/app", StaticFiles(directory="web", html=True), name="web")


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")


@app.get("/health", tags=["meta"], summary="Health check")
def health() -> Dict[str, Any]:
    return {"status": "ok", "llm": llm_enabled()}


@app.post("/api/index/build", tags=["meta"], summary="Build or rebuild TF-IDF index")
def build_index() -> Dict[str, Any]:
    stats = retriever.build_index()
    return {"ok": True, **stats}


@app.get("/api/meta/options", tags=["meta"], summary="List available topics/qtypes/difficulties")
def list_options() -> Dict[str, Any]:
    return retriever.get_options()


@app.post("/api/session/start", response_model=StartSessionResponse, tags=["session"], summary="Start a session")
def start_session(req: StartSessionRequest):
    cfg = req.config.model_dump() if req.config else None
    sid = store.create_session(req.profile.model_dump(), cfg)
    return StartSessionResponse(session_id=sid)


@app.get("/api/session/{session_id}/question", response_model=QuestionResponse, tags=["qa"], summary="Get next question")
def next_question(session_id: str, topic: str | None = None, force_new: bool = False):
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    pending = store.get_pending_question(session_id)
    if pending and not force_new:
        return QuestionResponse(**pending)

    profile = sess.get("profile", {})
    cfg = store.get_config(session_id)
    role = str(profile.get("role", "")).strip() or "data science"
    last = (sess.get("history") or [])[-1] if (sess.get("history")) else None
    query = role
    if topic:
        query += f" {topic}"
    if last:
        query += f" {last.get('question_text','')}"

    filters = cfg.get("filters") if isinstance(cfg, dict) else None
    hits = retriever.search(query=query, top_k=5, filters=filters)
    if not hits:
        raise HTTPException(500, "No questions available")
    top = hits[0]
    q = QuestionResponse(
        session_id=session_id,
        question_id=top["doc_id"],
        question_text=top["text"],
        source=top.get("source"),
        topic=top.get("topic"),
        qtype=top.get("qtype"),
        difficulty=top.get("difficulty"),
    )
    store.set_pending_question(session_id, q.model_dump())
    return q


@app.post("/api/session/{session_id}/answer", response_model=AnswerResponse, tags=["qa"], summary="Submit answer and get feedback")
def submit_answer(session_id: str, req: AnswerRequest):
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")

    pending = store.get_pending_question(session_id)
    if pending:
        q = QuestionResponse(**pending)
    else:
        profile = sess.get("profile", {})
        role = str(profile.get("role", "")).strip() or "data science"
        hits = retriever.search(query=role, top_k=5)
        if not hits:
            raise HTTPException(500, "No questions available")
        top = hits[0]
        q = QuestionResponse(
            session_id=session_id,
            question_id=top["doc_id"],
            question_text=top["text"],
            source=top.get("source"),
            topic=top.get("topic"),
        )

    cfg = store.get_config(session_id)
    filters = cfg.get("filters") if isinstance(cfg, dict) else None
    ctx = retriever.search(q.question_text, top_k=5, filters=filters)

    # Get scoring method from config (default: 'hybrid')
    scoring_method = cfg.get("scoring_method", "hybrid") if isinstance(cfg, dict) else "hybrid"
    model_weight = cfg.get("model_weight", 0.7) if isinstance(cfg, dict) else 0.7

    # Generate feedback based on selected method
    if llm_enabled():
        # LLM has highest priority if enabled
        prompt = (
            "You are an interview coach. Evaluate the candidate answer using the rubric.\n"
            f"Question: {q.question_text}\n"
            f"Answer: {req.answer_text}\n"
            f"Context (related topics): {[c['text'] for c in ctx]}\n"
            "Return JSON with: overall_score (1..5), breakdown{content_relevance,technical_accuracy,communication_clarity,structure_star}, strengths[], improvements[], tips[]."
        )
        fb = generate_feedback_with_llm(prompt) or hybrid_feedback(q.question_text, req.answer_text, ctx, model_weight=model_weight)
    elif scoring_method == "model":
        # Model-only scoring
        fb = model_feedback(q.question_text, req.answer_text, ctx)
    elif scoring_method == "hybrid":
        # Hybrid scoring (default): combines model + heuristic
        fb = hybrid_feedback(q.question_text, req.answer_text, ctx, model_weight=model_weight)
    else:
        # Fallback to heuristic
        fb = heuristic_feedback(req.answer_text, ctx)

    store.append_interaction(
        session_id=session_id,
        question_id=q.question_id,
        question_text=q.question_text,
        answer_text=req.answer_text,
        feedback=fb,
    )
    store.clear_pending_question(session_id)
    remaining = store.get_config(session_id).get("remaining")
    return AnswerResponse(session_id=session_id, question_id=q.question_id, feedback=fb, remaining=remaining)


@app.get("/api/session/{session_id}/progress", response_model=ProgressResponse, tags=["session"], summary="Get progress and history")
def progress(session_id: str):
    pr = store.progress(session_id)
    return ProgressResponse(session_id=session_id, total_q=pr["total_q"], avg_score=pr["avg_score"], history=pr["history"])


@app.post("/api/session/start_mock", response_model=StartSessionResponse, tags=["session"], summary="Start a fixed-length mock interview")
def start_mock(role: str, total_questions: int = 5, topic: str | None = None, qtype: str | None = None, difficulty: str | None = None):
    profile = {"role": role, "skills": [], "name": None, "resume_text": None}
    cfg = {"mode": "mock", "total_questions": total_questions, "filters": {"topic": topic, "qtype": qtype, "difficulty": difficulty}}
    sid = store.create_session(profile, cfg)
    return StartSessionResponse(session_id=sid)


@app.get("/api/session/{session_id}/summary", response_model=SummaryResponse, tags=["session"], summary="Get session summary (averages and top suggestions)")
def summary(session_id: str):
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    pr = store.progress(session_id)
    hist = pr["history"]
    # Aggregate breakdowns
    dims = ["content_relevance", "technical_accuracy", "communication_clarity", "structure_star"]
    sums = {d: 0.0 for d in dims}
    n = 0
    strengths: Dict[str, int] = {}
    improvements: Dict[str, int] = {}
    for h in hist:
        fb = (h.get("feedback") or {})
        br = fb.get("breakdown") or {}
        if all(isinstance(br.get(d), (int, float)) for d in dims):
            for d in dims:
                sums[d] += float(br[d])
            n += 1
        for s in fb.get("strengths", []) or []:
            strengths[s] = strengths.get(s, 0) + 1
        for im in fb.get("improvements", []) or []:
            improvements[im] = improvements.get(im, 0) + 1
    avg_breakdown = {d: round(sums[d] / n, 2) if n else 0.0 for d in dims}
    strengths_top = [k for k, _ in sorted(strengths.items(), key=lambda x: -x[1])[:5]]
    improvements_top = [k for k, _ in sorted(improvements.items(), key=lambda x: -x[1])[:5]]

    return SummaryResponse(
        session_id=session_id,
        completed=store.is_complete(session_id),
        total_q=pr["total_q"],
        avg_score=pr["avg_score"],
        avg_breakdown=avg_breakdown,
        strengths_top=strengths_top,
        improvements_top=improvements_top,
    )


@app.get("/api/session/{session_id}/export", tags=["session"], summary="Export session history as JSON")
def export_history(session_id: str) -> Dict[str, Any]:
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return sess
