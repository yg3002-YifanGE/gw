from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Profile(BaseModel):
    name: Optional[str] = None
    role: str = Field(..., description="Target job role, e.g., 'Data Scientist'")
    skills: List[str] = Field(default_factory=list)
    resume_text: Optional[str] = None


class FilterOptions(BaseModel):
    topic: Optional[str] = Field(default=None, description="Topic filter, e.g., 'Machine Learning' or 'Deep Learning'")
    qtype: Optional[str] = Field(default=None, description="Question type, e.g., 'ml' | 'dl' | 'technical' | 'behavioral'")
    difficulty: Optional[str] = Field(default=None, description="Difficulty level, 'easy' | 'medium' | 'hard'")


class SessionConfig(BaseModel):
    mode: Optional[str] = Field(default="free", description="'free' (default) or 'mock' for fixed-length interview")
    total_questions: Optional[int] = Field(default=None, ge=1, description="Total questions for mock interview")
    filters: Optional[FilterOptions] = None
    scoring_method: Optional[str] = Field(
        default="hybrid", 
        description="Scoring method: 'heuristic' (rule-based), 'model' (BERT only), or 'hybrid' (default, combines both)"
    )
    model_weight: Optional[float] = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Weight for model score in hybrid mode (0.0-1.0, default 0.7 means 70% model, 30% heuristic)"
    )


class StartSessionRequest(BaseModel):
    profile: Profile
    config: Optional[SessionConfig] = Field(default=None)


class StartSessionResponse(BaseModel):
    session_id: str


class QuestionResponse(BaseModel):
    session_id: str
    question_id: str
    question_text: str
    source: Optional[str] = None
    topic: Optional[str] = None
    qtype: Optional[str] = None
    difficulty: Optional[str] = None


class AnswerRequest(BaseModel):
    answer_text: str


class ScoreBreakdown(BaseModel):
    content_relevance: float
    technical_accuracy: float
    communication_clarity: float
    structure_star: float


class Feedback(BaseModel):
    overall_score: float
    breakdown: ScoreBreakdown
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    session_id: str
    question_id: str
    feedback: Feedback
    remaining: Optional[int] = None


class ProgressResponse(BaseModel):
    session_id: str
    total_q: int
    avg_score: Optional[float]
    history: List[Dict[str, Any]]


class SummaryResponse(BaseModel):
    session_id: str
    completed: bool
    total_q: int
    avg_score: Optional[float]
    avg_breakdown: Dict[str, float]
    strengths_top: List[str]
    improvements_top: List[str]
