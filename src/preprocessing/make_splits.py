import json
import random
from pathlib import Path

random.seed(22)

def split_jsonl(
    in_jsonl: str,
    out_dir: str,
    train_ratio=0.8,
    val_ratio=0.1
):
    with open(in_jsonl, 'r') as f:
        data = [json.loads(l) for l in f]

    random.shuffle(data)
    n = len(data)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        out_path = Path(out_dir) / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in items:
                f.write(json.dumps(ex) + "\n")
        print(f"{split_name}: wrote {len(items)} -> {out_path}")

def main():
    print("Splitting character dataset into train/val/test...")

    split_jsonl(
        in_jsonl="",
        out_dir="",
        train_ratio=0.8,
        val_ratio=0.1
    )

    print("Dataset splitting complete.")


if __name__ == "__main__":
    main()