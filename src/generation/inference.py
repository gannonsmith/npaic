import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.amp import autocast
import torch
from tqdm import tqdm
from src.util.build_prompt import build_prompt

BASE_MODEL_PATH = "models/lora_adapters/arthur_morgan" # TODO update to finetuned
TEST_DATA_PATH = "data/summarized_splits/dialogue_pairs_test_summarized.jsonl"
OUTPUT_PATH = "results/predictions/finetuned_model/predictions.jsonl"
BATCH_SIZE = 8

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def batch_iterator(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    test_data = list(load_jsonl(TEST_DATA_PATH))
    print(f"Test data loaded: {len(test_data)} examples")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for batch in tqdm(batch_iterator(test_data, BATCH_SIZE),
                          total=(len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE,
                          desc="Generating..."):
            prompts = [build_prompt(ex) for ex in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            for i, (ex, prompt) in enumerate(zip(batch, prompts)):
                decoded = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                prediction = decoded[len(prompt):].strip()

                outfile.write(json.dumps({
                    "mission": ex.get("mission", ""),
                    "context": ex.get("context", ""),
                    "speaker": ex.get("speaker", ""),
                    "utterance": ex.get("utterance", ""),
                    "response_speaker": ex.get("response_speaker", ""),
                    "gold_response": ex.get("response", ""),
                    "gold_response_action": ex.get("gold_response_action", "none"),
                    "predicted_response": prediction
                }) + "\n")

    print(f"Wrote baseline LLM predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

