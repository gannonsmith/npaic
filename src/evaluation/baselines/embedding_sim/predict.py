import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

TRAIN_FILE = "src/data/processed/rdr2/dialogue_pairs_train.jsonl"
TEST_FILE = "src/data/processed/rdr2/dialogue_pairs_test.jsonl"
OUTPUT_FILE = "src/data/baselines/embedding_sim/predictions.jsonl"


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_retrieval_corpus(train_data, model, device):
    """
    Build an embedding index from training contexts + utterances.
    Returns embeddings tensor and responses list.
    """
    corpus_texts = []
    responses = []

    for ex in train_data:
        text = f"{ex['context']} {ex['speaker']}: {ex['utterance']}"
        corpus_texts.append(text)
        responses.append(ex["response"])

    print(f"Encoding {len(corpus_texts)} training examples for retrieval...")
    corpus_embeddings = model.encode(
        corpus_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
        batch_size=64,
    )
    return corpus_embeddings, responses


def predict_retrieval(train_data, test_data, model, corpus_embeddings, responses, device="cuda"):
    """
    For each test example, find the most similar training example
    and return its response as the prediction.
    """
    predictions = []

    for ex in tqdm(test_data, desc="Retrieving"):
        query_text = f"{ex['context']} {ex['speaker']}: {ex['utterance']}"
        query_emb = model.encode(query_text, convert_to_tensor=True, device=device)
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        best_idx = int(torch.argmax(cos_scores))
        best_response = responses[best_idx]

        predictions.append({
            "mission": ex.get("mission", ""),
            "context": ex.get("context", ""),
            "speaker": ex.get("speaker", ""),
            "utterance": ex.get("utterance", ""),
            "response_speaker": ex.get("response_speaker", ""),
            "gold_response": ex.get("response", ""),
            "gold_response_action": ex.get("gold_response_action", "none"),
            "predicted_response": best_response
        })

    return predictions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Load data
    train_data = load_jsonl(TRAIN_FILE)
    test_data = load_jsonl(TEST_FILE)

    # Build retrieval corpus from training set
    corpus_embeddings, responses = build_retrieval_corpus(train_data, model, device)

    # Predict on test set
    predictions = predict_retrieval(train_data, test_data, model, corpus_embeddings, responses, device)

    # Save predictions
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in predictions:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(predictions)} retrieval baseline predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
