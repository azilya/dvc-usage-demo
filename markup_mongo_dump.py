import json
import os
import sys
from glob import glob

from sklearn.model_selection import train_test_split

folder = os.getcwd()


def main(run="bio_markup"):
    result = []
    for fname in glob(f"{folder}/dump/*.jsonl"):
        with open(fname) as fin:
            for line in fin:
                result.append(json.loads(line))

    labels = [r["sentiment_label"] for r in result]

    train, test, _, test_labels = train_test_split(
        result, labels, test_size=0.2, stratify=labels, random_state=42
    )
    test, dev, _, _ = train_test_split(
        test, test_labels, test_size=0.5, stratify=test_labels, random_state=42
    )
    assert (
        set([t for ex in train for t in ex["labels"]])
        == set([t for ex in test for t in ex["labels"]])
        == set([t for ex in dev for t in ex["labels"]])
    )

    for name, split in zip(["dev", "train", "test"], [dev, train, test]):
        with open(os.path.join(folder, "data", run, f"{name}.jsonl"), "w") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run = sys.argv[1]
    os.makedirs(run, exist_ok=True)
    main(run)
