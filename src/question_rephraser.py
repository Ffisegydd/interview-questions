import random
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType
from consts import Level, Question
import logging

REPHRASE_TEMPLATE = """
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


@component
class ReduceClusters:
    def __init__(self, model: ModelType) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialising instance")

        self.pipeline = Pipeline()

        self.pipeline.add_component(
            "rephraser-prompt",
            PromptBuilder(template=REPHRASE_TEMPLATE),  # type: ignore
        )
        self.pipeline.add_component("rephraser-generator", create_llm_model(model))

        self.pipeline.connect("rephraser-prompt", "rephraser-generator")

        self.reduced_clusters = []

    @component.output_types(questions=List[Question])
    def run(
        self,
        clusters: List[List[Question]],
        level: Level = Level.BEGINNER,
        max_cluster_size: int = 5,
    ):
        self.logger.info("Beginning run")
        for cluster in clusters:
            topics = list(set([q.topic for q in cluster]))
            sub_topics = list(set([q.sub_topic for q in cluster]))
            if len(sub_topics) > 1:
                # This is unlikely to happen, so logging if it does
                self.logger.warning(
                    f"Length of sub_topics in cluster is greater than 1. \nQuestions:\n{'\n'.join([f"    {q.question} ({q.sub_topic})" for q in cluster])}"
                )

            if len(cluster) > max_cluster_size:
                # This may be too many questions, so reducing the size
                self.logger.warning(
                    f"Cluster bigger than 5 ({len(cluster)}).\nQuestions:\n{'\n'.join([f"    {q.question} ({q.sub_topic})" for q in cluster])}"
                )
                cluster = random.choices(cluster, k=5)

            if len(cluster) == 1:
                self.reduced_clusters.append(cluster[0])
            else:
                reduced_cluster = self.pipeline.run(
                    {
                        "rephraser-prompt": {
                            "topics": topics,
                            "sub_topics": sub_topics,
                            "level": level,
                            "questions": cluster,
                        }
                    }
                )
                self.reduced_clusters.append(
                    Question(
                        question=reduced_cluster["rephraser-generator"]["replies"][0],
                        topic=topics[0],
                        sub_topic=sub_topics[0],
                    )
                )
        self.logger.info(f"# of questions rephrased: {len(self.reduced_clusters)}")
        return {"questions": self.reduced_clusters}


if __name__ == "__main__":
    questions = [
        "How do you handle missing data in a dataset?",
        "Is missing data a problem in a dataset?",
    ]
