from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Level(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Question(BaseModel):
    topic: str
    sub_topic: str
    question: str


class EmbeddedQuestion(Question):
    question_embedding: List[float]


class AnsweredQuestion(Question):
    answers: List[str]


class FullQuestion(AnsweredQuestion):
    follow_ups: List[str]


class Output(BaseModel):
    topic: str
    sub_topic: str
    question: str
    answers: List[str]
    follow_ups: List[str]


class CaseStudy(BaseModel):
    topic: str
    sub_topic: str
    text: str
    # context: str
    questions: Optional[str] = None
    answers: Optional[str] = None
