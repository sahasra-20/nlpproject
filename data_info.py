import argparse
import csv
from collections import Counter
from pathlib import Path


def show_data_info(csv_path: str) -> None:
    data_path = Path(csv_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Put data.csv in this folder or pass --csv PATH."
        )

    with data_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        rows = list(reader)

    print(f"Number of samples: {len(rows)}")
    print(f"Column names: {columns}")

    for column in ("crop", "intent"):
        if column not in columns:
            print(f"\nUnique values in {column}: column not found")
            continue

        value_counts = Counter(
            row[column].strip()
            for row in rows
            if row.get(column) and row[column].strip()
        )
        print(f"\nUnique values in {column} ({len(value_counts)} total, showing top 25):")
        for value, count in value_counts.most_common(25):
            print(f"- {value}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show basic dataset information.")
    parser.add_argument(
        "--csv",
        default="data.csv",
        help="Path to the dataset CSV file. Defaults to data.csv.",
    )
    args = parser.parse_args()

    show_data_info(args.csv)
