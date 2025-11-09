AI Interview Coach: FastAPI + RAG + Rubric Feedback

Overview
- Interactive interview practice API with rubric-based feedback
- RAG retrieval over Kaggle interview questions (TF‑IDF + cosine)
- Optional LLM feedback via OpenAI; heuristic fallback without a key
- Dockerized service exposing REST API + simple front-end UI

What’s Included
- Filters: topic / qtype (ml|dl|technical|behavioral) / difficulty (easy|medium|hard)
- Mock interview: fixed question count with remaining counter + summary
- Export: download full session JSON
- Front-end UI at `/app/` (dropdowns auto-populated from `/api/meta/options`)

Key Endpoints
- `GET /health` — Health check
- `POST /api/index/build` — (Re)build local TF‑IDF index from `kaggle_data`
- `GET /api/meta/options` — List available topics/qtypes/difficulties
- `POST /api/session/start` — Start a session with optional filters
- `POST /api/session/start_mock` — Start a fixed-length mock interview
- `GET /api/session/{session_id}/question` — Get next question; `force_new=true` to skip pending
- `POST /api/session/{session_id}/answer` — Submit an answer, get structured feedback
- `GET /api/session/{session_id}/progress` — Session progress and history
- `GET /api/session/{session_id}/summary` — Averages and top strengths/improvements
- `GET /api/session/{session_id}/export` — Export full session JSON

Quick Start (Docker)
- Build: `docker build -t ai-interview-coach .`
- One‑liner run: `bash docker_run.sh`
- Docs: `http://localhost:8000/docs`
- Front‑end UI: `http://localhost:8000/app/`

Quick Start (Local Python)
1) `pip install -r requirements.txt`
2) `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
3) `http://localhost:8000/docs` and `http://localhost:8000/app/`

Environment Variables
- `DATA_DIR` — Path to the Kaggle dataset (default: `../kaggle_data`)
- `INDEX_DIR` — Path to store TF‑IDF index (default: `./data/index`)
- `OPENAI_API_KEY` — If set, enables LLM‑enhanced feedback

Notes
- If no `OPENAI_API_KEY` is provided, the service uses a deterministic heuristic evaluator that scores answers by topical relevance and STAR structure.
- The dataset parser ingests:
  - `deeplearning_questions.csv`
  - `1. Machine Learning Interview Questions`
  - `2. Deep Learning Interview Questions`
  You can extend `rag/ingest.py` to parse more sources as needed.
