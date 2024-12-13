from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Level(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Result:
    topic: str
    sub_topic: str
    question: str
    question_embedding: Optional[List[float]] = None
    answers: Optional[List[str]] = None
    follow_ups: Optional[List[str]] = None
