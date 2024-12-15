from typing import List
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from models import create_llm_model, ModelType

sub_topic_generation_model = create_llm_model(ModelType.REMOTE)

SUB_TOPIC_TEMPLATE = """
You are writing a list of sub-topics that are relevant to the a given topic.
Your sub-topics should be clear and concise.
Thinks strategically, and don't write sub-topics that are too granular.
Write a series of sub-topics below, with one sub-topic per line.
You must not write the same sub-topic twice.
You must not include any extra conversation, you must only write sub-topics.
You must not include any numbers or bullet points at the start of each line.
You are an expert in the topic {{ topic }}.
Your topic to write sub-topics for is {{ topic }}.
You should write a maximum of {{ num_sub_topics }} sub-topics."""

sub_topic_prompt_builder = PromptBuilder(template=SUB_TOPIC_TEMPLATE)  # type: ignore

sub_topic_generation_pipeline = Pipeline()

sub_topic_generation_pipeline.add_component(
    "sub-topic-prompt", sub_topic_prompt_builder
)
sub_topic_generation_pipeline.add_component(
    "sub-topic-generator", sub_topic_generation_model
)

sub_topic_generation_pipeline.connect("sub-topic-prompt", "sub-topic-generator")


def generate_sub_topics(topic: str, num_sub_topics: int = 10) -> List[str]:
    results = sub_topic_generation_pipeline.run(
        {
            "sub-topic-prompt": {
                "topic": topic,
                "num_sub_topics": num_sub_topics,
            }
        }
    )
    sub_topics = results["sub-topic-generator"]["replies"][0].split("\n")

    return sub_topics


if __name__ == "__main__":
    key_points = generate_sub_topics(topic="machine learning engineering")
    print(key_points)
