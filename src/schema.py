from pydantic import BaseModel, Field

class LectureEvaluation(BaseModel):
    factual_consistency_score: float = Field(..., description="Score from 0.0 to 1.0 checking notes against source transcript.")
    hallucination_detected: bool
    missing_critical_terms: list[str]
