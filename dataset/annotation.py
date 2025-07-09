from pathlib import Path
import csv
import numpy as np

from typing import Dict, List, Union, Tuple, Set


def read_as_tags(filename: Union[str, Path]) -> Tuple[Dict[str, Set[str]], Set[str]]:
    labels = set()
    tags = {}
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            fn = row['filename']
            label = row['event_label']

            if fn not in tags:
                tags[fn] = {label}
            else:
                tags[fn].add(label)

            labels.add(label)

    return tags, labels


def read_fsd50k_tags(filename: Union[str, Path], use_mids: bool = True) -> Tuple[Dict[str, List[str]], List[str]]:
    col = 'mids' if use_mids else 'labels'
    labels = set()
    tags = {}

    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f)

        for row in reader:
            fn_id = row['fname']
            fn_tags = set(row[col].split(','))
            labels.update(fn_tags)
            tags[fn_id] = sorted(fn_tags)

    return tags, sorted(labels)


def generate_matrix(filenames: List[Path], tags: Dict[str, Set[str]], labels: List[str]) -> np.ndarray:
    mat = np.zeros((len(filenames), len(labels)))

    label_ind = {label: k for k, label in enumerate(labels)}

    for k, fn in enumerate(filenames):
        for label in tags[fn.stem]:
            if label not in label_ind:
                continue

            mat[k, label_ind[label]] = 1

    return mat
