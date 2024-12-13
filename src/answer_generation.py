from typing import List
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from consts import Result
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

answer_prompt_builder = PromptBuilder(template=ANSWER_TEMPLATE)  # type: ignore
answer_generation_model = create_llm_model(ModelType.REMOTE)

answer_generation_pipeline = Pipeline()

answer_generation_pipeline.add_component("answer-prompt", answer_prompt_builder)
answer_generation_pipeline.add_component("answer-generator", answer_generation_model)

answer_generation_pipeline.connect("answer-prompt", "answer-generator")


def generate_answer_key_points(result: Result) -> List[str]:
    results = answer_generation_pipeline.run(
        {
            "answer-prompt": {
                "topic": result.topic,
                "sub_topic": result.sub_topic,
                "question": result.question,
            }
        }
    )
    key_points = results["answer-generator"]["replies"][0].split("\n")

    return key_points


if __name__ == "__main__":
    question = Result(
        topic="machine learning engineering",
        sub_topic="algorithms",
        question="What are some key considerations when selecting a machine learning algorithm for a given problem?",
    )
    key_points = generate_answer_key_points(question)
    print(key_points)
