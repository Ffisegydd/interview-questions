import os
import json
import csv
from pathlib import Path
import re

import pandas as pd


def clean_lines(lines):
    return [re.sub(r"^[\s*-]+|[\s*-]+$", "", line) for line in lines if line]


def json_to_csv(json_file):
    # Initialize a list to hold all data
    df = pd.read_json(json_file, orient="records")

    df["follow_ups"] = df["follow_ups"].apply(clean_lines)
    df["answers"] = df["answers"].apply(clean_lines)
    df["follow_ups"] = df["follow_ups"].apply(lambda x: "\n".join(x))
    df["answers"] = df["answers"].apply(lambda x: "\n".join(x))

    csv_location = Path(json_file).with_suffix(".csv")

    df.to_csv(csv_location, index=False, quoting=0)


if __name__ == "__main__":
    result_dir = Path(__file__).parent / ".." / "results"
    print(result_dir)
    for json_file in result_dir.glob("*.json"):
        json_to_csv(json_file)
