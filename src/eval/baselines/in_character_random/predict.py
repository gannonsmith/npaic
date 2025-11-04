import json
import random
from pathlib import Path

ALL_LINES_FILE = "src/data/baselines/in_character_random/all_arthur_lines.json"
DIALOGUE_FILE = "src/data/processed/rdr2/dialogue_pairs_test.jsonl"
OUTPUT_FILE = "src/data/baselines/in_character_random/predictions.jsonl"


def predict(dialogues, all_lines):
    """
    Generate random baseline predictions for each dialogue.
    Selects a random line from all_lines regardless of context or character.
    """
    predictions = []
    for ex in dialogues:
        random_line = random.choice(all_lines)
        predictions.append({
            "mission": ex.get("mission", ""),
            "context": ex.get("context", ""),
            "speaker": ex.get("speaker", ""),
            "utterance": ex.get("utterance", ""),
            "response_speaker": ex.get("response_speaker", ""),
            "gold_response": ex.get("response", ""),
            "predicted_response": random_line
        })
    return predictions


def main() -> None:
    # Load all possible lines
    with open(ALL_LINES_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        # Ensure it's a flat list of strings
        all_lines = [l.strip() for l in all_lines if l.strip()]

    # Load dialogue pairs
    dialogues = []
    with open(DIALOGUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            dialogues.append(json.loads(line))

    print(f"Loaded {len(all_lines)} total lines and {len(dialogues)} dialogue pairs.")

    # Generate predictions
    preds = predict(dialogues, all_lines)

    # Save predictions to JSONL
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in preds:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Wrote {len(preds)} random baseline predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
