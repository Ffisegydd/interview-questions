import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from case_study_generation import generate_case_studies
from consts import CaseStudy
from sub_topic_generation import generate_sub_topics

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

other_logger_names = ["haystack.core.pipeline.pipeline", "httpx"]
for name in other_logger_names:
    logging.getLogger(name).setLevel(logging.CRITICAL)

state = []


def serialise_results(case_study: CaseStudy):
    state.append(
        {
            "text": case_study.text,
            "answers": case_study.answers,
            "topic": case_study.topic,
            "sub_topic": case_study.sub_topic,
        }
    )


TOPIC = "machine learning engineering"

results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
results_file = (
    results_dir
    / f"case_study_{TOPIC.replace(' ', '_')}_{datetime.now().isoformat()}.json"
)


def store_state():
    with open(results_file, "w") as f:  # Overwrite existing file
        json.dump(state, f, indent=4)


def main() -> None:
    logger.info("STARTING PIPELINE")
    # Note that due to du-dupe and rephrasing, you can't guarantee the number of case studies
    # at the end of the process
    NUM_CASE_STUDIES_PER_SUB_TOPIC = 5

    logger.info("Generating sub-topics")
    sub_topics = generate_sub_topics(topic=TOPIC)
    logger.info(f"# sub-topics generated: {len(sub_topics)}")
    logger.info(f"Sub-topics: \n{'\n'.join(sub_topics)}")
    input("Press Enter to continue, if happy with sub-topics...")

    logger.info("Generating case studies")
    raw_case_studies: List[CaseStudy] = []
    for sub_topic in sub_topics:
        logger.info(f"Generating case studies for sub-topic: {sub_topic}")
        sub_topic_case_studies = []

        while len(sub_topic_case_studies) < NUM_CASE_STUDIES_PER_SUB_TOPIC:
            case_study = generate_case_studies(topic=TOPIC, sub_topic=sub_topic)
            sub_topic_case_studies.append(case_study)

        raw_case_studies.extend(
            [
                CaseStudy(topic=TOPIC, sub_topic=sub_topic, text=text)
                for text in sub_topic_case_studies
            ]
        )
    logger.info(f"# case studies generated: {len(raw_case_studies)}")

    for study in raw_case_studies:
        serialise_results(study)

    logger.info("PIPELINE COMPLETE")
    store_state()


if __name__ == "__main__":
    main()
