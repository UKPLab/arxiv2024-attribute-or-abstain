import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def clean_and_normalize_data(
        annotation_data: pd.DataFrame,
) -> pd.DataFrame:
    # Remove rows where label is -1
    annotation_data = annotation_data.loc[annotation_data['label'] != -1]

    # Transform label values: 0 -> 0, 1 -> 0, 2 -> 1
    annotation_data['label'] = annotation_data['label'].apply(lambda x: 0 if x in [0, 1] else 1)

    return annotation_data


def main(
        annotations_csv_1_path: Path,
        annotations_csv_2_path: Path,
        n_classes: int
):
    results = {
        'class_distribution_1': None,
        'class_distribution_2': None,
        'n_annotations': 0,
        "cohen's kappa": 0
    }

    print(f'Loading annotations 1 from {annotations_csv_1_path}')
    annotations_1 = pd.read_csv(annotations_csv_1_path)
    print(f'Loading annotations 2 from {annotations_csv_2_path}')
    annotations_2 = pd.read_csv(annotations_csv_2_path)

    # Assert all ids are the same
    # Make id column by joining columns example_id, annotation_idx and sentence_idx
    annotations_1['id'] = annotations_1['example_id'].astype(str) + '_' + annotations_1['annotation_idx'].astype(str) + '_' + annotations_1['sentence_idx'].astype(str)
    annotations_2['id'] = annotations_2['example_id'].astype(str) + '_' + annotations_2['annotation_idx'].astype(str) + '_' + annotations_2['sentence_idx'].astype(str)

    # Assert all ids are the same
    assert all(annotations_1['id'] == annotations_2['id'])

    if n_classes == 3:
        # Assert that all values for label are in [-1, 0, 1, 2]
        assert all(annotations_1['label'].isin([-1, 0, 1, 2]))
        assert all(annotations_2['label'].isin([-1, 0, 1, 2]))
    elif n_classes == 2:
        assert all(annotations_1['label'].isin([-1, 0, 1]))
        assert all(annotations_2['label'].isin([-1, 0, 1]))
    else:
        raise ValueError

    # Assert all -1 labels are the same
    annotations_1_invalid = annotations_1['id'].loc[annotations_1['label'] == -1]
    annotations_2_invalid = annotations_2['id'].loc[annotations_2['label'] == -1]
    assert all(annotations_1_invalid == annotations_2_invalid)

    # Clean and normalize data
    if n_classes == 3:
        annotations_1 = clean_and_normalize_data(annotations_1)
        annotations_2 = clean_and_normalize_data(annotations_2)

    results['n_annotations'] = len(annotations_1)

    # Get label distributions as dictionaries
    class_distribution_1 = annotations_1['label'].value_counts().to_dict()
    class_distribution_2 = annotations_2['label'].value_counts().to_dict()
    results['class_distribution_1'] = class_distribution_1
    results['class_distribution_2'] = class_distribution_2

    # Compute cohen's kappa
    cohen_kappa = cohen_kappa_score(annotations_1['label'], annotations_2['label'])
    results["cohen's kappa"] = cohen_kappa

    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations_csv_1_path', type=Path)
    parser.add_argument('annotations_csv_2_path', type=Path)
    parser.add_argument('n_classes', type=int)
    args = parser.parse_args()
    main(args.annotations_csv_1_path, args.annotations_csv_2_path, args.n_classes)