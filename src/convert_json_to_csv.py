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

    columns_to_be_cleaned = ["answers", "follow_ups"]

    for column in columns_to_be_cleaned:
        try:
            df[column] = df[column].apply(clean_lines)
            df[column] = df[column].apply(lambda x: "\n".join(x))
        except Exception as e:
            print(f"Error cleaning column {column}: {e}")

    csv_location = Path(json_file).with_suffix(".csv")

    df.to_csv(csv_location, index=False, quoting=0)


if __name__ == "__main__":
    result_dir = Path(__file__).parent / ".." / "results"
    print(result_dir)
    for json_file in result_dir.glob("*.json"):
        json_to_csv(json_file)
