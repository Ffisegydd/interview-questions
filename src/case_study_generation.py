from typing import List, Optional
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType


CASE_STUDY_TEMPLATE = """
You are writing case studies for an interview.
Your case studies should be open-ended and thought-provoking.
Case studies should provide an initial problem as context, and lead the candidate through a series of questions.
You should also provide some guidance on what you'd expect to see in a good answer.
Guidance should be a series of bullet points, with one point per line. Do not include any punctuation at the start of your bullet points
All text should be clear and concise, and should not be overly verbose.
Questions should be written in a way to take the candidate through the problem-solving process. They should be answerable given the context and common sense.
You should write a maximum of five questions.
You must not write answers to questions.
You must not include any extra conversation, you must only write a single cast study.
Do not provide examples that discuss e-commerce.

You must write your case study using the template below.

Context:
<!-- Provide a brief context for the case study. -->

Questions
<1-- Provide a list of questions -->

Guidance
<!-- Provide guidance on what you'd expect to see in a good answer. -->

You are an expert interviewer in the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}.
Your case study must be on the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}.
Write a single case study below.
"""


case_study_generation_pipeline = Pipeline()

case_study_generation_pipeline.add_component(
    "case-study-prompt",
    PromptBuilder(template=CASE_STUDY_TEMPLATE),  # type: ignore
)
case_study_generation_pipeline.add_component(
    "case-study-generator", create_llm_model(ModelType.REMOTE)
)

case_study_generation_pipeline.connect("case-study-prompt", "case-study-generator")


def generate_case_studies(topic: str, sub_topic: Optional[str] = None) -> List[str]:
    results = case_study_generation_pipeline.run(
        {"case-study-prompt": {"topic": topic, "sub_topic": sub_topic}}
    )
    case_study = results["case-study-generator"]["replies"][0]

    return case_study


if __name__ == "__main__":
    case_study = generate_case_studies(
        topic="machine learning engineering", sub_topic="model privacy"
    )
    print(case_study)
