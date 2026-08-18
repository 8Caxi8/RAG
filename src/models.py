import uuid
from typing import List
from pydantic import BaseModel, ConfigDict, Field


class MinimalSource(BaseModel):
    """A minimal reference to a location inside a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a known answer yet."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    model_config = ConfigDict(populate_by_name=True)
    question_id: str
    question: str = Field(alias="question_str")
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for a single question, with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results produced by the student's system for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Search results with generated answers, for a dataset."""

    search_results: List[MinimalAnswer]  # type: ignore[assignment]
