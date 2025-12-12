import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.knowledge.retriever import KnowledgeGraphRetriever


BASE_MODEL_PATH = "models/base/qwen2.5-3b"

INPUT_DIR = "data/splits"
OUTPUT_DIR = "data/summarized_splits"

TRAIN_FILE = f"{INPUT_DIR}/dialogue_pairs_train.jsonl"
VAL_FILE = f"{INPUT_DIR}/dialogue_pairs_val.jsonl"
TEST_FILE = f"{INPUT_DIR}/dialogue_pairs_test.jsonl"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model

def run_model(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

def summarize_memory(context, tokenizer, model):
    if not context.strip():
        return "<no memory>"
    prompt = (
        "Summarize the following game dialogue in 2-3 sentences. "
        "Focus on emotional tone, motivation, and important facts.\n\n"
        f"{context}\n\nSummary:"
    )
    return run_model(tokenizer, model, prompt)

def summarize_knowledge(example, retriever, tokenizer, model):
    facts = retriever.get_relevant_facts(
        mission=example.get("mission"),
        context=example.get("context"),
        speaker=example["speaker"],
        target=example["response_speaker"]
    )

    if not facts:
        return "<no relevant facts>"
    
    fact_text = "\n".join(facts)

    prompt = (
        "Summarize the following knowledge facts into a compact form that "
        "is useful for guiding the character's next line.\n\n"
        f"{fact_text}\n\nSummary:"
    )

    return run_model(tokenizer, model, prompt)

def process_splits(path, output_path, retriever, tokenizer, model):
    data = load_jsonl(path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating summaries for: {path}")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(data):
            memory_summary = summarize_memory(
                ex["context"], tokenizer, model
            )
            kg_summary = summarize_knowledge(
                ex, retriever, tokenizer, model
            )

            ex_out = dict(ex)
            ex_out["memory_summary"] = memory_summary
            ex_out["knowledge_summary"] = kg_summary

            out_f.write(json.dumps(ex_out + "\n"))
            out_f.flush()
    
    print(f"Saved summarized dataset to {output_path}")

def main():
    tokenizer, model = load_model()
    # tokenizer, model = None, None

    retriever = KnowledgeGraphRetriever()

    process_splits(
        TRAIN_FILE,
        f"{OUTPUT_DIR}/dialogue_pairs_train_summarized.jsonl",
        retriever, tokenizer, model
    )
    process_splits(
        VAL_FILE,
        f"{OUTPUT_DIR}/dialogue_pairs_val_summarized.jsonl",
        retriever, tokenizer, model
    )
    process_splits(
        TEST_FILE,
        f"{OUTPUT_DIR}/dialogue_pairs_test_summarized.jsonl",
        retriever, tokenizer, model
    )

if __name__ == "__main__":
    main()
