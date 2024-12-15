from typing import List
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from consts import Question
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


def generate_answer_key_points(result: Question) -> List[str]:
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
    question_body = """Title: Building a Production-Ready Machine Learning Model for Fraud Detection
Problem: A financial services company wants to build a production-ready machine learning model to detect fraudulent transactions in real-time. The company has a large dataset of historical transaction data, but lacks the expertise and infrastructure to develop and deploy machine learning models at scale.
1. What are the key steps involved in building a production-ready machine learning model for fraud detection?
2. How can the financial services company ensure that their model is accurate and reliable?
3. What are some common challenges in developing and deploying machine learning models, and how can these be addressed?
4. How can the financial services company integrate their machine learning model with their existing IT infrastructure and workflows?
5. What is ML Ops, and how can it help the financial services company to streamline their machine learning development and deployment processes?
"""
    # question_body = "What are some key considerations when selecting a machine learning algorithm for a given problem?"

    question = Question(
        topic="machine learning engineering",
        sub_topic="algorithms",
        question=question_body,
    )
    key_points = generate_answer_key_points(question)
    print(key_points)
