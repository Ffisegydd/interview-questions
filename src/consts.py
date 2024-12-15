from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Level(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Question:
    topic: str
    sub_topic: str
    question: str
    question_embedding: Optional[List[float]] = None
    answers: Optional[List[str]] = None
    follow_ups: Optional[List[str]] = None


class CaseStudy(BaseModel):
    topic: str
    sub_topic: str
    text: str
    # context: str
    questions: Optional[str] = None
    answers: Optional[str] = None
