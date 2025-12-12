from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

# INPUT_PATH = "src/data/baselines/random_line/predictions.jsonl"
# INPUT_PATH = "src/data/baselines/in_character_random/predictions.jsonl"
INPUT_PATH = "src/data/baselines/embedding_sim/predictions.jsonl"

smooth = SmoothingFunction().method1

def compute_bleu(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        # BLEU expects lists of tokens
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
        scores.append(score)
    return scores

def main():
    # Example usage
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    # Suppose you have baseline outputs:
    preds = [ex["predicted_response"] for ex in data]
    refs = [ex["gold_response"] for ex in data]

    bleu_scores = compute_bleu(preds, refs)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU: {avg_bleu:.3f}")

if __name__ == "__main__":
    main()
