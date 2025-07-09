
from pathlib import Path
import csv


def read_tags(filename: str | Path, split: str | None = None) -> tuple[dict[str, list[str]], list[str]]:
    mids = set()
    tags = {}

    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_mids = row['mids'].split(',')
            mids.update(file_mids)

            if split is not None and row['split'] != split:
                continue

            fn = row['fname']
            tags[fn] = file_mids

    return tags, sorted(mids)


def _test():
    fn = '/home/mnu/Documents/data/fsd50k/FSD50K.ground_truth/dev.csv'
    tags, labels = read_tags(fn, split='val')

    print(len(tags), len(labels))

if __name__ == '__main__':
    _test()
