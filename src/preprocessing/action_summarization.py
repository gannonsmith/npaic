import json
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from data.api.gemini.client import GeminiClient

def build_batch_prompt(batch: List[Dict]) -> str:
    """
    Create the prompt for Gemini summarization.
    """
    lines = "\n".join(
        [f"{i+1}. {ex['response']}" for i, ex in enumerate(batch)]
    )
    return (
        "Summarize each NPC response below as a short verb phrase "
        "describing the action the player is told or encouraged to take. "
        "If no action is suggested, write 'none'. "
        "Return only a numbered list of answers matching the inputs. \n\n"
        f"{lines}"
    )

def add_action_summary_batched(
    in_path: str,
    out_path: str,
    batch_size: int=10,
    model: str = "gemini-2.5-flash",
    api_key_var: str = "SUPER_SECRET_KEY"
):
    gemini = GeminiClient(
        model=model,
        api_key_var=api_key_var,
        batch_size=batch_size
    )

    with open(in_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    data = [d for d in data if d.get("gold_response_action", "none") == "none"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    updated = []
    for i in tqdm(range(0, len(data), batch_size), desc="Summarizing (batched)"):
        batch = data[i:i + batch_size]
        prompt = build_batch_prompt(batch)

        try:
            output = gemini.ask(prompt)
            actions = [a.strip() for a in output.split("\n") if a.strip()]

            for j, ex in enumerate(batch):
                ex["gold_response_action"] = (
                    actions[j] if j < len(actions) else "none"
                )
                updated.append(ex)
        
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            for ex in batch:
                ex["gold_response_action"] = "failed action summary"
                updated.append(ex)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in updated:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Wrote {len(updated)} examples with action summaries to {out_path}")

def retry_failed_action_summaries(
    input_path: str,
    output_path: str,
    batch_size: int = 10,
    sleep_time: float = 3.0
):
    """
    Retry examples where gold_response_action is "failed action summary"
    """

    with open(input_path, "r", encoding="utf-8") as f:
        full_data = [json.loads(line) for line in f]
    
    to_retry = [
        ex for ex in full_data
        if ex.get("gold_response_action", "") == "failed action summary"
    ]

    print(f"Retrying {len(to_retry)} examples...")

    if not to_retry:
        print("No retries needed.")
        return
    
    gemini = GeminiClient(model="gemini-2.5-flash", batch_size=batch_size)
    updated = []

    for i in tqdm(range(0, len(to_retry), batch_size), desc="Retrying"):
        batch = to_retry[i:i + batch_size]
        prompt = build_batch_prompt(batch)

        try:
            output = gemini.ask(prompt)
            actions = [a.strip() for a in output.split("\n") if a.strip()]

            for j, ex in enumerate(batch):
                ex["gold_response_action"] = (
                    actions[j] if j < len(actions) else "none"
                )
                updated.append(ex)

        except Exception as e:
            print(f"Failed batch {i}: {e}")
            for ex in batch:
                ex["gold_response_action"] = "failed action summary"
                updated.append(ex)

        time.sleep(sleep_time)

    # Merge back into original data
    updated_dict = {
        json.dumps({k: ex[k] for k in ex if k != "gold_response_action"}):
            ex["gold_response_action"]
        for ex in updated
    }

    merged = []
    for ex in full_data:
        key = json.dumps({k: ex[k] for k in ex if k != "gold_response_action"})
        if key in updated_dict:
            ex["gold_response_action"] = updated_dict[key]
        merged.append(ex)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in merged:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved merged dataset → {output_path}")

def remove_numbered_list(input_path: str, output_path: str):
    """
    Remove leading '1. ', '2. ', etc. from Gemini responses.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        full_data = [json.loads(line) for line in f]

    for d in full_data:
        val = d.get("gold_response_action", "")
        if len(val) > 3 and val[1] == ".":
            d["gold_response_action"] = val[3:]

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in full_data:
            f.write(json.dumps(ex) + "\n")

    print(f"Cleaned numbered summaries -> {output_path}")

def main():
    add_action_summary_batched(
        in_path="data/splits/arthur_dialogue_pairs.jsonl",
        out_path="data/splits/arthur_dialogue_pairs_action.jsonl",
        batch_size=10,
        model="gemini-2.5-flash",
        api_key_var="YOU_WISH"
    )

    retry_failed_action_summaries(
        input_path="data/splits/arthur_dialogue_pairs_action.jsonl",
        output_path="data/splits/arthur_dialogue_pairs_action.jsonl",
        batch_size=10,
        sleep_time=3.0
    )

    remove_numbered_list(
        input_path="data/splits/arthur_dialogue_pairs_action.jsonl",
        output_path="data/splits/arthur_dialogue_pairs_action.jsonl"
    )

    print("Action summarization pipeline complete!")

if __name__ == "__main__":
    main()