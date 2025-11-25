import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    doc_id: str
    text: str
    source: str
    topic: str | None = None


class TfIdfRetriever:
    def __init__(self, data_dir: str, index_dir: str) -> None:
        self.data_dir = data_dir
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.vectorizer_path = os.path.join(self.index_dir, "vectorizer.pkl")
        self.matrix_path = os.path.join(self.index_dir, "matrix.npy")
        self.meta_path = os.path.join(self.index_dir, "meta.json")

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: np.ndarray | None = None
        self.meta: List[Dict[str, Any]] = []

    def _read_dataset(self) -> List[Document]:
        docs: List[Document] = []
        # CSV with potential BOM in header
        csv_path = os.path.join(self.data_dir, "deeplearning_questions.csv")
        if os.path.exists(csv_path):
            import csv

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("ID") or row.get("\ufeffID") or ""
                    rid = str(rid).strip()
                    text = (row.get("DESCRIPTION") or row.get("\ufeffDESCRIPTION") or "").strip()
                    if not text:
                        continue
                    doc_id = f"dl-{rid}" if rid else f"dl-auto-{abs(hash(text))%10_000_000}"
                    docs.append(Document(doc_id=doc_id, text=text, source=os.path.basename(csv_path), topic="Deep Learning"))

        # Text files: numbered interview questions
        for fname, topic in [
            ("1. Machine Learning Interview Questions", "Machine Learning"),
            ("2. Deep Learning Interview Questions", "Deep Learning"),
        ]:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line or line.lower().startswith("s.no"):
                            continue
                        parts = [p.strip() for p in line.split("\t", 1)]
                        qtext = parts[-1] if parts else line
                        # Short, readable id like "1-24"
                        short_prefix = fname.split(".")[0]
                        doc_id = f"{short_prefix}-{i}"
                        docs.append(Document(doc_id=doc_id, text=qtext, source=fname, topic=topic))

        return docs

    def build_index(self) -> Dict[str, Any]:
        docs = self._read_dataset()
        
        # Filter out very short documents first
        valid_docs = [d for d in docs if len(d.text.split()) >= 2]
        
        if len(valid_docs) < 2:
            raise ValueError(f"Not enough valid documents (need at least 2, got {len(valid_docs)})")
        
        texts = [d.text for d in valid_docs]
        
        def heuristics(doc: Document) -> Dict[str, str]:
            text = (doc.text or "").lower()
            # qtype heuristic
            if (doc.topic or "").lower().startswith("deep"):
                qtype = "dl"
            elif (doc.topic or "").lower().startswith("machine"):
                qtype = "ml"
            else:
                qtype = "technical"
            # difficulty heuristic based on token count and advanced terms
            length = len(text.split())
            adv_terms = sum(1 for t in ["derive", "prove", "convergence", "regularization", "eigen", "manifold", "bayesian", "hessian"] if t in text)
            if length <= 6 and adv_terms == 0:
                difficulty = "easy"
            elif length <= 14 and adv_terms <= 1:
                difficulty = "medium"
            else:
                difficulty = "hard"
            return {"qtype": qtype, "difficulty": difficulty}

        meta = [{
            "doc_id": d.doc_id,
            "source": d.source,
            "topic": d.topic,
            "text": d.text,
            **heuristics(d)
        } for d in valid_docs]
        
        # Use TfidfVectorizer with min_df to avoid empty vocabulary
        # min_df=1 means a word must appear in at least 1 document (no filtering)
        # max_features limits vocabulary size for efficiency
        # ngram_range helps with short texts
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,  # Accept words that appear in at least 1 document
            max_features=5000,  # Limit vocabulary size
            ngram_range=(1, 2)  # Use unigrams and bigrams for better matching
        )
        matrix = vectorizer.fit_transform(texts)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        np.save(self.matrix_path, matrix.toarray().astype(np.float32))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.vectorizer = vectorizer
        self.matrix = matrix.toarray().astype(np.float32)
        self.meta = meta
        return {"docs_indexed": len(valid_docs), "total_docs": len(docs), "filtered": len(docs) - len(valid_docs)}

    def _ensure_loaded(self) -> None:
        if self.vectorizer is not None and self.matrix is not None and self.meta:
            return
        if not (os.path.exists(self.vectorizer_path) and os.path.exists(self.matrix_path) and os.path.exists(self.meta_path)):
            self.build_index()
        else:
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.matrix = np.load(self.matrix_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def search(self, query: str, top_k: int = 5, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self.vectorizer is not None and self.matrix is not None
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]

        # Build candidate indices, applying filters by meta
        idxs_all = np.argsort(-sims)
        selected = []
        for idx in idxs_all:
            m = self.meta[idx]
            if filters:
                topic_ok = (filters.get("topic") is None) or (str(m.get("topic")) == filters.get("topic"))
                qtype_ok = (filters.get("qtype") is None) or (str(m.get("qtype")) == filters.get("qtype"))
                diff_ok = (filters.get("difficulty") is None) or (str(m.get("difficulty")) == filters.get("difficulty"))
                if not (topic_ok and qtype_ok and diff_ok):
                    continue
            selected.append(idx)
            if len(selected) >= max(top_k, 20):  # grab extra for safety
                break
        idxs = selected[:top_k]
        results: List[Dict[str, Any]] = []
        for idx in idxs:
            m = self.meta[idx]
            results.append({
                "doc_id": m["doc_id"],
                "text": m["text"],
                "source": m.get("source"),
                "topic": m.get("topic"),
                "qtype": m.get("qtype"),
                "difficulty": m.get("difficulty"),
                "score": float(sims[idx]),
            })
        return results
