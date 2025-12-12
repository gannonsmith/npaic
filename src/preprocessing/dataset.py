import json
from pathlib import Path
from typing import Dict, List

def count_lines(script: Dict[str, List[Dict]]):
    """
    Count how many lines each character speaks.
    """
    counter = {}
    for sec, lines in script.items():
        for entry in lines:
            if entry['type'] == 'scripted line':
                char = entry["character"]
                counter[char] = counter.get(char, 0) + 1
    return counter

def generate_pairs_per_character(in_jsonl: str, out_jsonl: str, character: str):
    """
    Extract only examples where that character is the *response speaker*.
    """
    character = character.lower()
    filtered = []

    with open(in_jsonl, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if ex["response_speaker"].lower() == character:
                filtered.append(ex)

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, 'w') as f:
        for ex in filtered:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(filtered)} dialogue pairs for '{character}' to {out_jsonl}")

def generate_all_responses(script: Dict[str, List[Dict]], out_jsonl: str):
    """
    Write all spoken-line responses into a flat JSONL list.
    """
    lines = []
    for sec, entries in script.items():
        for entry in entries:
            if entry['type'] == 'scripted line':
                lines.append(entry['line'])

    with open(out_jsonl, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    
    print(f"Collected {len(lines)} lines' to {out_jsonl}")

def generate_all_responses_per_character(
    script: Dict[str, List[Dict]],
    out_jsonl: str,
    character: str
):
    lines = []
    for sec, entries in script.items():
        for entry in entries:
            if entry['type'] == 'scripted line' and entry['character'] == character:
                lines.append(entry['line'])

    with open(out_jsonl, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    
    print(f"Wrote {len(lines)} lines for {character} to {out_jsonl}")

def main():
    print("Loading structured script JSON...")
    script_path = Path("data/processed/script.json")

    with open(script_path, "r") as f:
        script = json.load(f)


    print("Counting dialogue lines per character...")
    stats = count_lines(script)

    stats_path = Path("data/processed/line_count.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved character stats to {stats_path}")

    print("Extracting training pairs...")
    generate_pairs_per_character(
        in_jsonl="data/processed/dialogue_pairs.jsonl",
        out_jsonl="data/splits/arthur_dialogue_pairs.jsonl",
        character="Arthur Morgan",
    )

    print("Saving all spoken lines...")
    generate_all_responses(
        script=script,
        out_jsonl="data/processed/all_lines.jsonl"
    )

    print("Saving all per character lines...")
    generate_all_responses_per_character(
        script=script,
        out_jsonl="data/processed/all_arthur_lines.jsonl",
        character="Arthur Morgan"
    )

    print("Dataset utilities complete!")

if __name__ == "__main__":
    main()