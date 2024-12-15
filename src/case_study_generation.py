import logging
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from consts import CaseStudy
from models import create_llm_model, ModelType


CASE_STUDY_TEMPLATE = """
You are writing case studies for an interview.
Your case studies should be open-ended and thought-provoking.
Case studies should provide an initial problem as context, and lead the candidate through a series of questions.
All text should be clear and concise, and should not be overly verbose.
Questions should be written in a way to take the candidate through the problem-solving process. They should be answerable given the context and a knowledgeable candidate.
You should write a maximum of five questions.
You must not write answers to questions.
You must not include any extra conversation, you must only write a single cast study.
Domains of interest include: business, government, and defence.

You should output your case study in the following format:

Context

<Your context here>

Questions

<Your questions here>

You are an expert interviewer in the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}.
Your case study must be on the topic {{ topic }}{{ (" and sub-topic " + sub_topic) if sub_topic else "" }}.
Write a single case study below.
"""


@component
class GenerateCaseStudies:
    def __init__(self, model: ModelType) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialising instance")
        self.pipeline = Pipeline()

        self.pipeline.add_component(
            "case-study-prompt",
            PromptBuilder(template=CASE_STUDY_TEMPLATE),  # type: ignore
        )
        self.pipeline.add_component(
            "case-study-generator",
            create_llm_model(model),
        )

        self.pipeline.connect("case-study-prompt", "case-study-generator")

    @component.output_types(questions=List[CaseStudy])
    def run(
        self,
        topic: str,
        sub_topics: List[str],
        num_case_studies: int,
        **kwargs,
    ):
        self.logger.info("Beginning run")

        case_studies: List[CaseStudy] = []
        for sub_topic in sub_topics:
            self.logger.debug(
                f"Generating case study contexts for sub-topic: {sub_topic}"
            )
            num_generated_for_sub_topic = 0
            while num_generated_for_sub_topic < num_case_studies:
                results = self.pipeline.run(
                    {
                        "case-study-prompt": {
                            "topic": topic,
                            "sub_topic": sub_topic,
                        }
                    }
                )
                context = results["case-study-generator"]["replies"][0]

                case_studies.append(
                    CaseStudy(topic=topic, sub_topic=sub_topic, context=context)
                )
                num_generated_for_sub_topic += 1

        self.logger.info(f"# of raw questions generated: {len(case_studies)}")
        return {"case_studies": case_studies}
