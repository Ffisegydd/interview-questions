from typing import List, Optional
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType
from consts import Level

question_generation_model = create_llm_model(ModelType.REMOTE)

QUESTION_TEMPLATE = """
You are writing interview questions on the topic for others to use.
Your interview questions should be open-ended and thought-provoking.
{% if level == "advanced" %}
You should ask questions that throw curve balls and are very difficult.
{% endif %}
Write a series of questions below, with one question per line.
You must not write the same question twice.
You must not write answers to questions.
You must not include any numbers at the start of questions.
You must not include any extra conversation, you must only write questions.
Your questions should be aimed at a {{ level }} practitioner.
You are an expert interviewer in the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}."""

question_prompt_builder = PromptBuilder(template=QUESTION_TEMPLATE)  # type: ignore

question_generation_pipeline = Pipeline()

question_generation_pipeline.add_component("question-prompt", question_prompt_builder)
question_generation_pipeline.add_component(
    "question-generator", question_generation_model
)

question_generation_pipeline.connect("question-prompt", "question-generator")


def generate_questions(
    topic: str, sub_topic: Optional[str] = None, level: Level = Level.BEGINNER
) -> List[str]:
    results = question_generation_pipeline.run(
        {
            "question-prompt": {
                "topic": topic,
                "sub_topic": sub_topic,
                "level": level.value,
            }
        }
    )
    questions = results["question-generator"]["replies"][0].split("\n")

    return questions


if __name__ == "__main__":
    questions = generate_questions(
        topic="machine learning engineering",
        sub_topic="machine learning algorithms",
        level=Level.BEGINNER,
    )
    print(questions)
