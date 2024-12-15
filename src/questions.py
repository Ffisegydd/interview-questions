import json
import logging
from datetime import datetime
from pathlib import Path

from haystack import Pipeline

from answer_generation import GenerateAnswers
from cluster_similar_documents import ClusterQuestions
from consts import FullQuestion, Level
from followup_generation import GenerateFollowups
from models import ModelType
from question_generation import (
    GenerateQuestions,
)
from question_rephraser import ReduceClusters
from sub_topic_generation import generate_sub_topics

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

other_logger_names = ["haystack.core.pipeline.pipeline", "httpx"]
for name in other_logger_names:
    logging.getLogger(name).setLevel(logging.CRITICAL)

state = []


def serialise_results(result: FullQuestion):
    state.append(
        {
            "question": result.question,
            "answers": result.answers,
            "topic": result.topic,
            "sub_topic": result.sub_topic,
            "follow_ups": result.follow_ups,
        }
    )


results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / f"{datetime.now().isoformat()}.json"


def store_state():
    with open(results_file, "w") as f:  # Overwrite existing file
        json.dump(state, f, indent=4)


def create_pipeline(model_type: ModelType = ModelType.REMOTE_CHEAP) -> Pipeline:
    logger.info("Creating pipeline")

    question_generation_pipeline = Pipeline()

    question_generation_pipeline.add_component(
        "generate-questions", GenerateQuestions(model_type)
    )
    question_generation_pipeline.add_component(
        "cluster-questions", ClusterQuestions("mxbai-embed-large")
    )  # type: ignore
    question_generation_pipeline.add_component(
        "reduce-clusters", ReduceClusters(model_type)
    )  # type: ignore
    question_generation_pipeline.add_component(
        "generate-answers", GenerateAnswers(model_type)
    )  # type: ignore
    question_generation_pipeline.add_component(
        "generate-followups",
        GenerateFollowups(model_type),  # type: ignore
    )

    question_generation_pipeline.connect(
        "generate-questions.questions", "cluster-questions.questions"
    )
    question_generation_pipeline.connect(
        "cluster-questions.clusters", "reduce-clusters.clusters"
    )
    question_generation_pipeline.connect(
        "reduce-clusters.questions", "generate-answers.questions"
    )
    question_generation_pipeline.connect(
        "generate-answers.questions", "generate-followups.questions"
    )
    return question_generation_pipeline


if __name__ == "__main__":
    logger.info("Starting script")
    model_type = ModelType.REMOTE_CHEAP
    topic = "machine learning"
    num_sub_topics = 3
    level = Level.BEGINNER.value
    num_questions_per_sub_topic = 20
    max_cluster_size = 5
    pipeline = create_pipeline(model_type)

    sub_topics = generate_sub_topics(topic, num_sub_topics=num_sub_topics)
    logger.info(f"Sub topics: {sub_topics}")
    logger.info("Running pipeline")
    questions = pipeline.run(
        {
            "generate-questions": {
                "topic": topic,
                "sub_topics": sub_topics,
                "level": level,
                "num_questions": 20,
            },
            "reduce-clusters": {
                "max_cluster_size": max_cluster_size,
                "level": level,
            },
        }
    )
    for question in questions["generate-followups"]["questions"]:
        serialise_results(question)
    store_state()
