import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from src.util.build_prompt import build_prompt

BASE_MODEL_PATH = "models/base/qwen2.5-3b"
TEST_DATA_PATH = "data/splits/dialogue_pairs_test.jsonl"
OUTPUT_PATH = "results/predictions/base_model/predictions.jsonl"

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    test_data = load_jsonl(TEST_DATA_PATH)

    print(f"Test data loaded")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for ex in tqdm(test_data, desc="Generating..."):
            prompt = build_prompt(ex)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,
                top_p=0.8
            )

            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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

