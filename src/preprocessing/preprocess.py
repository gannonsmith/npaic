import json
from pathlib import Path
from typing import List, Dict

SECTIONS_FILE = "data/processed/sections.txt"
SCRIPT_FILE = "data/raw/cleaned_script.txt"

def read_sections() -> List[str]:
    with open(SECTIONS_FILE, 'r') as f:
        lines = [li.strip() for li in f.readlines()]
    return lines

def read_script() -> List[str]:
    with open(SCRIPT_FILE, 'r') as f:
        lines = [li.strip() for li in f.readlines()]
    return list(filter(None, lines))

def process_into_sections() -> Dict[str, List[str]]:
    """
    Groups the raw script into sections based on SECTIONS_FILE.
    """
    sections = read_sections()
    script = read_script()

    script_by_section = {}
    current_section = ""
    buffer = []

    for line in script:
        if line in sections:
            script_by_section[current_section] = buffer
            buffer = []
            current_section = line
        else:
            buffer.append(line)

    script_by_section[current_section] = buffer
    return script_by_section

def populate_script(sectioned: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
    """
    Converts raw text lines into {type, character?, lines} dicts.
    """
    info = {}

    for sec, lines in sectioned.items():
        info[sec] = []
        for line in lines:
            entry = {}

            if line.startswith('['):
                entry['type'] = 'action'
                entry['line'] = line

            elif line.startswith('"'):
                entry['type'] = 'journal'
                entry['line'] = line

            elif line.startswith('<'):
                entry['type'] = 'context'
                entry['line'] = line

            else:
                # expected: CHARACTER: dialogue text
                parts = line.split(':')
                character, text = parts[0], ":".join(parts[1:]).strip()
                entry['type'] = 'scripted line'
                entry['character'] = character
                entry['line'] = text

            info[sec].append(entry)

    return info


def preprocess_dialogue(
    data: Dict[str, List[Dict]], mission_name: str, window_len: int
):
    """
    Convert structured mission data into dialogue context->response pairs.
    """
    processed = []
    context = [] 
    prev_speaker, prev_line = None, None

    for entry in data:
        etype = entry["type"]

        if etype == "action":
            context.append(f"<action> {entry['line']} </action>")
            continue

        if etype == "journal":
            context.append(f"<journal> {entry['line']} </journal>")
            continue

        if etype == "context":
            context.append(f"<context> {entry['line']} </context>")
            continue

        if etype == "scripted line":
            speaker = entry["character"].strip()
            text = entry["line"].strip()

            # Create example if speaker changes
            if prev_speaker and prev_speaker != speaker:
                ex = {
                    "mission": mission_name,
                    "context": " ".join(context[-window_len:]),
                    "speaker": prev_speaker,
                    "utterance": prev_line,
                    "response_speaker": speaker,
                    "response": text
                }
                processed.append(ex)

            context.append(f"<{speaker}> {text} </{speaker}>")
            prev_speaker, prev_line = speaker, text

    return processed

def preprocess_all_missions(raw_jsonl_path: str, out_jsonl_path: str, window_len: int):
    """
    raw_json_path: JSON with {MissionName: [diaglogue entries...]}
    Produces a JSONL file of dialogue pairs.
    """
    with open(raw_jsonl_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_entries = []
    for mission_name, lines in data.items():
        all_entries.extend(
            preprocess_dialogue(lines, mission_name, window_len)
        )

    out_path = Path(out_jsonl_path)
    with open(out_path, 'w') as f:
        for ex in all_entries:
            f.write(json.dumps(ex) + '\n')

    print(f"Wrote {len(all_entries)} dialogue pairs to {out_path}")

def main():
    print("Processing raw script into sections...")
    sections = process_into_sections()

    print("Populationg structered script")
    script = populate_script(sections)

    output_script = Path("data/processed/script.json")
    output_script.parent.mkdir(parents=True, exist_ok=True)

    with open(output_script, "w") as f:
        json.dump(script, f, indent=4)
    
    print(f"Saved structured script -> {output_script}")

    print(f"Preprocessing missions into dialogue pairs (window={10})...")
    preprocess_all_missions(
        raw_jsonl_path=str(output_script)
        out_jsonl_path="data/processed/dialogue_pairs.jsonl",
        window_len=10
    )

    print("Preprocessing complete")


if __name__ == "__main__":
    main()