from pydantic import BaseModel, Field

class LectureEvaluation(BaseModel):
    factual_consistency_score: float = Field(
        ...,
        description="Score from 0.0 to 1.0 measuring how accurately the notes reflect the source transcript."
    )
    summary_quality_score: float = Field(
        ...,
        description="Score from 0.0 to 1.0 rating the clarity, structure, and completeness of the generated study notes."
    )
    hallucination_detected: bool = Field(
        ...,
        description="True if any claim in the notes or flashcards cannot be grounded in the source transcript."
    )
    missing_critical_terms: list[str] = Field(
        ...,
        description="List of important concepts present in the transcript that were omitted from the generated outputs."
    )
    key_concepts_covered: list[str] = Field(
        ...,
        description="List of the most important concepts from the transcript that were successfully captured in the outputs."
    )
