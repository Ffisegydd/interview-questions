import logging
import re
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from consts import AnsweredQuestion, Question
from models import create_llm_model, ModelType

ANSWER_TEMPLATE = """
You are writing key points that a good answer to a question should include.
Write a series of key points below, with one key point per line.
You MUST be clear and concise, and MUST NOT be overly verbose.
Key points should help someone mark a candidate in an interview.
Your points must be in a sensible order for someone reading them sequentially.
You should provide a maximum of 8 key points to be aware of.
You must ensure each key point is sufficiently different.
You must not include any extra conversation, you must only write key points.
You must not include any kind of bullet points, numbers, or punctuation at the start of each line.
You are an expert in the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}. 
Your question is {{ question }}"""


@component
class GenerateAnswers:
    def __init__(self, model: ModelType) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialising instance")
        self.pipeline = Pipeline()

        self.pipeline.add_component(
            "answer-prompt",
            PromptBuilder(template=ANSWER_TEMPLATE),  # type: ignore
        )
        self.pipeline.add_component(
            "answer-generator",
            create_llm_model(model),
        )

        self.pipeline.connect("answer-prompt", "answer-generator")

    @component.output_types(questions=List[AnsweredQuestion])
    def run(self, questions: List[Question]):
        self.logger.info("Beginning run")
        output: List[AnsweredQuestion] = []
        for idx, question in enumerate(questions):
            if idx % 20 == 0:
                self.logger.debug(f"Processing question {idx + 1}/{len(questions)}")
            results = self.pipeline.run(
                {
                    "answer-prompt": {
                        "topic": question.topic,
                        "sub_topic": question.sub_topic,
                        "question": question.question,
                    }
                }
            )
            answers = self._process_answer(results["answer-generator"]["replies"][0])
            output.append(
                AnsweredQuestion(
                    topic=question.topic,
                    sub_topic=question.sub_topic,
                    question=question.question,
                    answers=answers,
                )
            )

        return {"questions": output}

    def _process_answer(self, answer_text: str) -> List[str]:
        answers = answer_text.split("\n")
        answers = [
            re.sub(r"^[\s*-]+|[\s*-]+$", "", answer) for answer in answers if answer
        ]
        return answers


if __name__ == "__main__":
    question_body = """Title: Building a Production-Ready Machine Learning Model for Fraud Detection
Problem: A financial services company wants to build a production-ready machine learning model to detect fraudulent transactions in real-time. The company has a large dataset of historical transaction data, but lacks the expertise and infrastructure to develop and deploy machine learning models at scale.
1. What are the key steps involved in building a production-ready machine learning model for fraud detection?
2. How can the financial services company ensure that their model is accurate and reliable?
3. What are some common challenges in developing and deploying machine learning models, and how can these be addressed?
4. How can the financial services company integrate their machine learning model with their existing IT infrastructure and workflows?
5. What is ML Ops, and how can it help the financial services company to streamline their machine learning development and deployment processes?
"""
    # question_body = "What are some key considerations when selecting a machine learning algorithm for a given problem?"
