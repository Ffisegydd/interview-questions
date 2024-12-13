import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from answer_generation import generate_answer_key_points
from cluster_similar_documents import cluster_documents
from consts import Level, Result
from followup_generation import generate_followups
from question_generation import generate_questions
from question_rephraser import rephrase_questions
from sub_topic_generation import generate_sub_topics

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(logging.Logger.manager.loggerDict)
other_logger_names = ["haystack.core.pipeline.pipeline", "httpx"]
for name in other_logger_names:
    logging.getLogger(name).setLevel(logging.CRITICAL)

state = []


def serialise_results(result: Result):
    state.append(
        {
            "question": result.question,
            "answers": result.answers,
            "topic": result.topic,
            "sub_topic": result.sub_topic,
            "follow_ups": result.follow_ups,
        }
    )


TOPIC = "data engineering"
LEVEL = Level.BEGINNER

results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
results_file = (
    results_dir / f"{TOPIC.replace(' ', '_')}_{datetime.now().isoformat()}.json"
)


def store_state():
    with open(results_file, "w") as f:  # Overwrite existing file
        json.dump(state, f, indent=4)


def main() -> None:
    logger.info("STARTING PIPELINE")
    # Note that due to du-dupe and rephrasing, you can't guarantee the number of questions
    # at the end of the process
    MIN_QUESTIONS_FOR_SUB_TOPIC = 20

    logger.info("Generating sub-topics")
    sub_topics = generate_sub_topics(topic=TOPIC)
    logger.info(f"# sub-topics generated: {len(sub_topics)}")
    logger.info(f"Sub-topics: {sub_topics}")

    logger.info("Generating questions")
    raw_questions: List[Result] = []
    for sub_topic in sub_topics:
        logger.info(f"Generating questions for sub-topic: {sub_topic}")
        sub_topic_questions = []

        while len(sub_topic_questions) < MIN_QUESTIONS_FOR_SUB_TOPIC:
            questions = generate_questions(
                topic=TOPIC, sub_topic=sub_topic, level=LEVEL
            )
            sub_topic_questions.extend(questions)

        raw_questions.extend(
            [
                Result(topic=TOPIC, sub_topic=sub_topic, question=q)
                for q in sub_topic_questions
            ]
        )
    logger.info(f"# questions generated: {len(raw_questions)}")
    logger.info("Clustering questions")
    clustered_questions = cluster_documents(raw_questions)
    logger.info(f"# clusters: {len(clustered_questions)}")

    for i, cluster in enumerate(clustered_questions.values()):
        if len(cluster) > 1:
            logger.info(
                f"Cluster of {len(cluster)} questions: {[q.question for q in cluster]}"
            )
            result = rephrase_questions(questions=cluster, level=LEVEL)[0]
        else:
            result = cluster[0]

        answers = generate_answer_key_points(
            result=result,
        )
        result.answers = answers

        follow_ups = generate_followups(result=result, level=LEVEL)
        result.follow_ups = follow_ups
        serialise_results(result=result)

        if i % 20 == 0:
            logger.info(f"Storing state at cluster {i} / {len(clustered_questions)}")
            store_state()

    logger.info("PIPELINE COMPLETE")
    store_state()


if __name__ == "__main__":
    main()
