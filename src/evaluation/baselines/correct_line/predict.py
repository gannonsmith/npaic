import json
import random
from pathlib import Path

TEST_FILE = "data/splits/dialogue_pairs_test.jsonl"
OUTPUT_FILE = "results/predictions/correct_line/predictions.jsonl"


def predict(dialogues):
    predictions = []
    for ex in dialogues:
        predictions.append({
            "mission": ex.get("mission", ""),
            "context": ex.get("context", ""),
            "speaker": ex.get("speaker", ""),
            "utterance": ex.get("utterance", ""),
            "response_speaker": ex.get("response_speaker", ""),
            "gold_response": ex.get("response", ""),
            "gold_response_action": ex.get("gold_response_action", "none"),
            "predicted_response": ex.get("response", "none"),
        })
    return predictions


def main() -> None:
    # Load dialogue pairs
    dialogues = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            dialogues.append(json.loads(line))

    # Generate predictions
    preds = predict(dialogues)

    # Save predictions to JSONL
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in preds:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Wrote {len(preds)}baseline predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
