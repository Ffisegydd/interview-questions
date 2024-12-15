import json
import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint

from haystack import Pipeline

from case_study_generation import GenerateCaseStudies
from consts import CaseStudy, Level
from models import ModelType
from sub_topic_generation import generate_sub_topics

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

other_logger_names = ["haystack.core.pipeline.pipeline", "httpx"]
for name in other_logger_names:
    logging.getLogger(name).setLevel(logging.CRITICAL)

state = []


def serialise_results(result: CaseStudy):
    state.append(
        {
            "context": result.context,
            "answers": result.answers,
            "topic": result.topic,
            "sub_topic": result.sub_topic,
            "questions": result.questions,
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
    # Note that due to du-dupe and rephrasing, you can't guarantee the number of questions
    # at the end of the process

    question_generation_pipeline = Pipeline()

    question_generation_pipeline.add_component(
        "generate-case-studies", GenerateCaseStudies(model_type)
    )
    # question_generation_pipeline.add_component(
    #     "cluster-questions", ClusterQuestions("mxbai-embed-large")
    # )  # type: ignore
    # question_generation_pipeline.add_component(
    #     "reduce-clusters", ReduceClusters(model_type)
    # )  # type: ignore
    # question_generation_pipeline.add_component(
    #     "generate-answers", GenerateAnswers(model_type)
    # )  # type: ignore
    # question_generation_pipeline.add_component(
    #     "generate-followups",
    #     GenerateFollowups(model_type),  # type: ignore
    # )

    # question_generation_pipeline.connect(
    #     "generate-case-studies.questions", "cluster-questions.questions"
    # )
    # question_generation_pipeline.connect(
    #     "cluster-questions.clusters", "reduce-clusters.clusters"
    # )
    # question_generation_pipeline.connect(
    #     "reduce-clusters.questions", "generate-answers.questions"
    # )
    # question_generation_pipeline.connect(
    #     "generate-answers.questions", "generate-followups.questions"
    # )
    return question_generation_pipeline


if __name__ == "__main__":
    logger.info("Starting script")
    model_type = ModelType.REMOTE_CHEAP
    topic = "machine learning"
    num_sub_topics = 1
    level = Level.BEGINNER.value
    num_case_studies_per_sub_topic = 5
    max_cluster_size = 5
    pipeline = create_pipeline(model_type)

    sub_topics = generate_sub_topics(topic, num_sub_topics=num_sub_topics)
    logger.info(f"Sub topics: {sub_topics}")
    logger.info("Running pipeline")
    case_studies = pipeline.run(
        {
            "generate-case-studies": {
                "topic": topic,
                "sub_topics": sub_topics,
                "num_case_studies": num_case_studies_per_sub_topic,
            }
        }
    )
    for case_study in case_studies["generate-case-studies"]["case_studies"]:
        serialise_results(case_study)
        pprint(case_study.model_dump())
    store_state()
