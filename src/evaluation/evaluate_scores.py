from bert_score import score
from src.evaluation.bleu import compute_bleu
import json

# INPUT_PATH = "src/data/baselines/random_line/predictions.jsonl"
# INPUT_PATH = "src/data/baselines/in_character_random/predictions.jsonl"
# INPUT_PATH = "src/data/baselines/embedding_sim/predictions.jsonl"

# INPUT_PATH = "results/predictions/base_model/predictions.jsonl"
# INPUT_PATH = "results/predictions/finetuned_model/predictions.jsonl"
INPUT_PATH = "results/predictions/correct_line/predictions.jsonl"

def main():
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    preds = [ex["predicted_response"] for ex in data]
    refs = [ex["gold_response"] for ex in data]

    # Compute BLEU
    bleu_scores = compute_bleu(preds, refs)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU: {avg_bleu:.3f}")

    # Compute BERTScore
    P, R, F1 = score(preds, refs, lang="en", verbose=True)
    print(f"Average BERTScore Precision: {P.mean().item():.3f}")
    print(f"Average BERTScore Recall: {R.mean().item():.3f}")
    print(f"Average BERTScore F1: {F1.mean().item():.3f}")

if __name__ == "__main__":
    main()