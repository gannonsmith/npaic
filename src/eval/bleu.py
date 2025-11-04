from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

smooth = SmoothingFunction().method1

def compute_bleu(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        # BLEU expects lists of tokens
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
        scores.append(score)
    return sum(scores) / len(scores)

# Example usage
with open("data/processed/dialogue_pairs.jsonl") as f:
    data = [json.loads(line) for line in f]

# Suppose you have baseline outputs:
preds = [ex["baseline_output"] for ex in data]
refs = [ex["response"] for ex in data]

avg_bleu = compute_bleu(preds, refs)
print(f"Average BLEU: {avg_bleu:.3f}")
