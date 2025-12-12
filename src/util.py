

def build_prompt(example):
    """
    Builds a consistent input prompt for the baseline.
    """
    return (
        f"### Context:\n{example['context']}\n\n"
        f"### {example['speaker']} says:\n{example['utterance']}\n\n"
        f"### Respond as {example['response_speaker']}:\n"
    )
