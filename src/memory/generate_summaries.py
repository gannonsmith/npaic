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

def run_model_batch(tokenizer, model, prompts, max_new_tokens=120):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9
        )
    
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return [out.strip() for out in decoded]

def summarize_memory(context, tokenizer, model):
    if not context.strip():
        return "<no memory>"
    prompt = (
        "Summarize the following game dialogue in 2-3 sentences. "
        "Focus on emotional tone, motivation, and important facts.\n\n"
        f"{context}\n\nSummary:"
    )
    return run_model(tokenizer, model, prompt)

def summarize_memory_batch(contexts, tokenizer, model):
    prompts = []
    for ctx in contexts:
        if not ctx.strip():
            prompts.append("<no memory>:\n\nRespond with N/A")
        else:
            prompts.append(
                "Summarize this dialogue in 2-3 sentences, focusing on the emotional tone, "
                "character motivations, and key facts:\n\n"
                f"{ctx}\n\n"
                "Summary:"
            )
    summaries = run_model_batch(tokenizer, model, prompts)
    return summaries

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
        "Summarize the following knowledge graph facts into a compact form that "
        "is useful for guiding the character's next line.\n\n"
        f"{fact_text}\n\nSummary:\n"
    )

    return run_model(tokenizer, model, prompt)

def summarize_knowledge_batch(examples, retriever, tokenizer, model):
    prompts = []
    for example in examples:
        facts = retriever.get_relevant_facts(
            mission=example.get("mission"),
            context=example.get("context"),
            speaker=example["speaker"],
            target=example["response_speaker"]
        )
        if not facts:
            prompts.append("<no relevant facts>\n\nRespond with N/A")
        else:
            fact_text = "\n".join(facts)
            prompts.append(
                f"You are Arthur Morgan from Red Dead Redemption 2. Based on these facts about your world, "
                "write 2-3 sentences describing your perspective and feelings.\n\n"
                f"Facts:\n{fact_text}\n\n"
                f"Arthur's perspective:"
            )

    summaries = run_model_batch(tokenizer, model, prompts)
    return summaries

def process_splits(path, output_path, retriever, tokenizer, model, batch_size=4):
    data = load_jsonl(path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating summaries for: {path}")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]

            contexts = [ex['context'] for ex in batch]
            memory_summaries = summarize_memory_batch(
                contexts, tokenizer, model
            )

            kg_summaries = summarize_knowledge_batch(
                batch, retriever, tokenizer, model
            )

            for ex, mem_sum, kg_sum in zip(batch, memory_summaries, kg_summaries):
                ex_out = dict(ex)
                ex_out["memory_summary"] = mem_sum
                ex_out["knowledge_summary"] = kg_sum

                out_f.write(json.dumps(ex_out) + "\n")
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
