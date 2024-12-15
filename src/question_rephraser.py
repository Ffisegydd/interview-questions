import random
from typing import List
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType
from consts import Level, Question
import logging


logger = logging.getLogger(__name__)

question_rephrase_model = create_llm_model(ModelType.REMOTE)

QUESTION_TEMPLATE = """
You need to summarise some questions into a single question.
If the questions are too broad and don't overlap, pick a subset to ask a single question.
You must not write the same question twice.
You must not write answers to questions.
You must not include any numbers at the start of questions.
You must not include any extra conversation, you must only write questions.
Your question should be aimed at a {{ level }} practitioner.
You are an expert interviewer in the topic {{ ", ".join(topics) }}{{ (" and sub-topics " + ", ".join(sub_topic)) if sub_topics else "" }}. 
Write a single summarised question below.
Your questions to summarise are:
{% for question in questions %}
- {{ question }}
{% endfor %}
"""

question_rephrase_prompt_builder = PromptBuilder(template=QUESTION_TEMPLATE)  # type: ignore

question_rephrase_pipeline = Pipeline()

question_rephrase_pipeline.add_component(
    "rephraser-prompt", question_rephrase_prompt_builder
)
question_rephrase_pipeline.add_component("rephraser-generator", question_rephrase_model)

question_rephrase_pipeline.connect("rephraser-prompt", "rephraser-generator")


def rephrase_questions(
    questions: List[Question],
    level: Level = Level.BEGINNER,
) -> List[Question]:
    topics = list(set([q.topic for q in questions]))
    sub_topics = list(set([q.sub_topic for q in questions]))
    if len(sub_topics) > 1:
        # This is unlikely to happen, so logging if it does
        logging.warning(
            f"Length of sub_topics in cluster is greater than 1. \nQuestions:\n{'\n'.join([f"    {q.question} ({q.sub_topic})" for q in questions])}"
        )

    if len(questions) > 5:
        # This may be too many questions, so reducing the size
        logger.warning(
            f"Length of questions is greater than 5 ({len(questions)}). Reducing the size.\nQuestions:\n{'\n'.join([f"    {q.question} ({q.sub_topic})" for q in questions])}"
        )
        questions = random.choices(questions, k=5)
    results = question_rephrase_pipeline.run(
        {
            "rephraser-prompt": {
                "topics": topics,
                "sub_topics": sub_topics,
                "level": level.value,
                "questions": questions,
            }
        }
    )
    rephrased_questions = results["rephraser-generator"]["replies"][0].split("\n")

    rephrased_questions = [
        Question(topic=topics[0], sub_topic=sub_topics[0], question=q)
        for q in rephrased_questions
    ]

    return rephrased_questions


if __name__ == "__main__":
    questions = [
        "How do you handle missing data in a dataset?",
        "Is missing data a problem in a dataset?",
    ]
    results = [
        Question(topic="machine learning engineering", sub_topic="data", question=q)
        for q in questions
    ]
    rephrased_questions = rephrase_questions(
        questions=results,
        level=Level.BEGINNER,
    )
    print(rephrased_questions)
