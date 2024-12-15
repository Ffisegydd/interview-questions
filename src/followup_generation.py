from typing import List
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType
from consts import Level, Question

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


def generate_followups(
    result: Question,
    level: Level = Level.BEGINNER,
) -> List[str]:
    results = followup_generation_pipeline.run(
        {
            "followup-prompt": {
                "topic": result.topic,
                "sub_topic": result.sub_topic,
                "question": result.question,
                "answers": result.answers,
                "level": level.value,
            }
        }
    )
    questions = results["followup-generator"]["replies"][0].split("\n")

    return questions


if __name__ == "__main__":
    questions = generate_followups(
        result=Question(
            topic="machine learning engineering",
            sub_topic="machine learning algorithms",
            question="What is the difference between supervised and unsupervised learning?",
            answers=[
                "Supervised learning is a type of machine learning where the model is trained on a labeled dataset.",
                "Unsupervised learning is a type of machine learning where the model is trained on an unlabeled dataset.",
            ],
        ),
        level=Level.BEGINNER,
    )
    print(questions)
