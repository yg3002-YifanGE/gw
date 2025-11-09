import json
import os
import time
import uuid
from typing import Dict, Any, List


class SessionStore:
    def __init__(self, path: str = "./data/sessions.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except Exception:
                self._store = {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def create_session(self, profile: Dict[str, Any], config: Dict[str, Any] | None = None) -> str:
        sid = str(uuid.uuid4())
        self._store[sid] = {
            "session_id": sid,
            "profile": profile,
            "history": [],  # list of {question_id, question_text, answer_text, feedback, timestamp}
            "config": self._normalize_config(config or {}),
        }
        self._save()
        return sid

    def get(self, session_id: str) -> Dict[str, Any] | None:
        return self._store.get(session_id)

    # Pending question lifecycle -------------------------------------------------
    def set_pending_question(self, session_id: str, q: Dict[str, Any]) -> None:
        rec = self._store.get(session_id)
        if not rec:
            return
        rec["pending_question"] = q
        self._save()

    def get_pending_question(self, session_id: str) -> Dict[str, Any] | None:
        rec = self._store.get(session_id)
        if not rec:
            return None
        return rec.get("pending_question")

    def clear_pending_question(self, session_id: str) -> None:
        rec = self._store.get(session_id)
        if not rec:
            return
        rec.pop("pending_question", None)
        self._save()

    def append_interaction(
        self,
        session_id: str,
        question_id: str,
        question_text: str,
        answer_text: str,
        feedback: Dict[str, Any],
    ) -> None:
        now = int(time.time())
        rec = self._store.get(session_id)
        if not rec:
            return
        rec["history"].append(
            {
                "question_id": question_id,
                "question_text": question_text,
                "answer_text": answer_text,
                "feedback": feedback,
                "ts": now,
            }
        )
        # If mock interview, decrement remaining
        cfg = rec.get("config", {})
        if cfg.get("mode") == "mock" and isinstance(cfg.get("remaining"), int):
            cfg["remaining"] = max(0, cfg.get("remaining", 0) - 1)
            rec["config"] = cfg
        self._save()

    def progress(self, session_id: str) -> Dict[str, Any]:
        rec = self._store.get(session_id)
        if not rec:
            return {"total_q": 0, "avg_score": None, "history": []}
        history: List[Dict[str, Any]] = rec.get("history", [])
        scores = [h.get("feedback", {}).get("overall_score") for h in history]
        scores = [s for s in scores if isinstance(s, (int, float))]
        avg = sum(scores) / len(scores) if scores else None
        return {"total_q": len(history), "avg_score": avg, "history": history}

    # Config helpers -------------------------------------------------------------
    def _normalize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        mode = cfg.get("mode") or "free"
        total = int(cfg.get("total_questions", 0) or 0)
        filters = cfg.get("filters") or {}
        norm = {
            "mode": mode,
            "total_questions": total if mode == "mock" and total > 0 else None,
            "remaining": total if mode == "mock" and total > 0 else None,
            "filters": {
                k: v for k, v in filters.items() if k in {"topic", "qtype", "difficulty"} and v
            },
        }
        return norm

    def get_config(self, session_id: str) -> Dict[str, Any]:
        rec = self._store.get(session_id) or {}
        return rec.get("config", {})

    def is_complete(self, session_id: str) -> bool:
        cfg = self.get_config(session_id)
        if cfg.get("mode") == "mock":
            return isinstance(cfg.get("remaining"), int) and cfg.get("remaining", 0) <= 0
        return False
