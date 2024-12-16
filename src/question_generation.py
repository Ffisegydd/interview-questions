import os
import logging
from typing import List

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from consts import Level, Question
from models import ModelType, create_llm_model


openai_api_key = os.getenv("OPENAI_API_KEY", None)

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


@component
class GenerateQuestions:
    def __init__(self, model: ModelType) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialising instance")
        self.pipeline = Pipeline()

        self.pipeline.add_component(
            "question-prompt",
            PromptBuilder(template=QUESTION_TEMPLATE),  # type: ignore
        )
        self.pipeline.add_component(
            "question-generator",
            create_llm_model(model),
        )

        self.pipeline.connect("question-prompt", "question-generator")

    @component.output_types(questions=List[Question])
    def run(
        self,
        topic: str,
        sub_topics: List[str],
        level: Level,
        num_questions: int,
        **kwargs,
    ):
        self.logger.info("Beginning run")

        questions: List[Question] = []
        for sub_topic in sub_topics:
            self.logger.debug(f"Generating questions for sub-topic: {sub_topic}")
            num_generated_for_sub_topic = 0
            while num_generated_for_sub_topic < num_questions:
                results = self.pipeline.run(
                    {
                        "question-prompt": {
                            "topic": topic,
                            "sub_topic": sub_topic,
                            "level": level,
                        }
                    }
                )
                replies = results["question-generator"]["replies"]

                for reply in replies:
                    for line in reply.split("\n"):
                        cleaned_text = self.clean_question(line)
                        question = Question(
                            topic=topic, sub_topic=sub_topic, question=cleaned_text
                        )
                        if (
                            question.question
                            and num_generated_for_sub_topic < num_questions
                        ):  # Check for empty lines
                            questions.append(question)
                            if num_generated_for_sub_topic % 20 == 0:
                                self.logger.debug(
                                    f"Added question {num_generated_for_sub_topic}/{num_questions} for sub-topic: {sub_topic}"
                                )
                            num_generated_for_sub_topic += 1
        self.logger.info(f"# of raw questions generated: {len(questions)}")
        return {"questions": questions}

    def clean_question(self, question: str) -> str:
        return question.strip()
