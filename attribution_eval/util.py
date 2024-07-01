from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

import pandas as pd


@dataclass
class AttributionInstance:
    """Defines the attributes of an attribution instance"""
    task_name: str
    example_id: str
    annotation_idx: int
    sentence_idx: int
    claim: str
    evidence: str
    label: int  # 1 (attributable) or 0 (not attributable)
    answer_type: str


def save_attribution_dataset_jsonl(
        dataset: List[AttributionInstance],
        path: Path
):
    # Save jsonl
    with open(path, "w") as f:
        for instance in dataset:
            f.write(json.dumps(instance.__dict__))
            f.write("\n")


def save_attribution_dataset_csv(
        dataset: List[AttributionInstance],
        path: Path
):
    columns = {
        'label': [],
        'claim': [],
        'evidence': [],
        'task_name': [],
        'example_id': [],
        'annotation_idx': [],
        'sentence_idx': [],
        'answer_type': []
    }
    for attribution_instance in dataset:
        for column_name in columns:
            columns[column_name].append(getattr(attribution_instance, column_name))
    df = pd.DataFrame(columns)
    df.to_csv(path, index=False)

def load_attribution_dataset_jsonl(
        path: Path
) -> List[AttributionInstance]:
    dataset = []
    with open(path) as f:
        for line in f:
            instance = AttributionInstance(**json.loads(line))
            dataset.append(instance)
    return dataset


def load_attribution_dataset_csv(
        path: Path,
        is_three_way_annotation: bool = True
) -> List[AttributionInstance]:
    df = pd.read_csv(path)
    # Iterate over the rows and create an AttributionInstance for each row
    dataset = []
    for idx, row in df.iterrows():
        if row['label'] != -1:
            label = row['label']
            if is_three_way_annotation:
                if label in [0, 1]:
                    label = 0
                else:
                    label = 1
            else:
                if label == 2:
                    raise ValueError(
                        'is_three_way_annotation set to False '
                        'but label is 2!'
                    )
            instance = AttributionInstance(
                task_name=row["task_name"],
                example_id=row["example_id"],
                annotation_idx=row["annotation_idx"],
                sentence_idx=row["sentence_idx"],
                claim=row["claim"],
                evidence=row["evidence"],
                label=label,
                answer_type=row["answer_type"]
            )
            dataset.append(instance)
    return dataset

