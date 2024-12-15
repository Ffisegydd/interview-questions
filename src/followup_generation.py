import logging
import re
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType
from consts import AnsweredQuestion, FullQuestion, Level, Question

followup_generation_model = create_llm_model(ModelType.REMOTE)

FOLLOWUP_TEMPLATE = """
You are writing interview questions for others to use.
Below is a question that you have previously written. 
There are also some notes on what a good answer may contain.
Your job is to write some follow-up questions that an interviewer can use.
Follow-up questions are used to dive deeper into the subject after an initial answer.
They must ask for more detail or nuance on the topic, that may not have been covered in the original question.
They may also ask for examples or applications of the topic.
They should not deviate too far from the original question.
You must not write answers to questions.
You must not include any numbers at the start of questions.
You must not include any extra conversation, you must only write questions.
Your questions should be aimed at a {{ level }} practitioner.
You are an expert in the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}. 

QUESTION:
{{ question }}

ANSWER NOTES:
{% for answer in answers %}
{{ answer }}
{% endfor %}

FOLLOW-UP QUESTIONS:
"""

question_prompt_builder = PromptBuilder(template=FOLLOWUP_TEMPLATE)  # type: ignore

followup_generation_pipeline = Pipeline()

followup_generation_pipeline.add_component("followup-prompt", question_prompt_builder)
followup_generation_pipeline.add_component(
    "followup-generator", followup_generation_model
)

followup_generation_pipeline.connect("followup-prompt", "followup-generator")


@component
class GenerateFollowups:
    def __init__(self, model: ModelType) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialising instance")
        self.pipeline = Pipeline()

        self.pipeline.add_component(
            "followup-prompt",
            PromptBuilder(template=FOLLOWUP_TEMPLATE),  # type: ignore
        )
        self.pipeline.add_component(
            "followup-generator",
            create_llm_model(model),
        )

        self.pipeline.connect("followup-prompt", "followup-generator")

    @component.output_types(questions=List[FullQuestion])
    def run(self, questions: List[AnsweredQuestion]):
        self.logger.info("Beginning run")
        output: List[FullQuestion] = []
        for idx, question in enumerate(questions):
            if idx % 20 == 0:
                self.logger.debug(f"Processing question {idx + 1}/{len(questions)}")
            results = self.pipeline.run(
                {
                    "followup-prompt": {
                        "topic": question.topic,
                        "sub_topic": question.sub_topic,
                        "question": question.question,
                    }
                }
            )
            followups = self._process_answer(
                results["followup-generator"]["replies"][0]
            )
            output.append(
                FullQuestion(
                    topic=question.topic,
                    sub_topic=question.sub_topic,
                    question=question.question,
                    answers=question.answers,
                    follow_ups=followups,
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
    pass
    # questions = generate_followups(
    #     result=Question(
    #         topic="machine learning engineering",
    #         sub_topic="machine learning algorithms",
    #         question="What is the difference between supervised and unsupervised learning?",
    #         answers=[
    #             "Supervised learning is a type of machine learning where the model is trained on a labeled dataset.",
    #             "Unsupervised learning is a type of machine learning where the model is trained on an unlabeled dataset.",
    #         ],
    #     ),
    #     level=Level.BEGINNER,
    # )
    # print(questions)
