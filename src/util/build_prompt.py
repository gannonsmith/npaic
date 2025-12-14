def build_prompt(example):
    return (
        "You are roleplaying as Arthur Morgan from Red Dead Redemption 2.\n"
        "Stay in character and respond naturally.\n\n"
        f"Mission:\n{example['mission']}\n\n"
        f"Conversation Memory:\n{example['memory_summary']}\n\n"
        f"Relevant Knowledge:\n{example['knowledge_summary']}\n\n"
        f"Dialogue:\n"
        f"{example['speaker']}: {example['utterance']}\n\n"
        f"{example['response_speaker']}:"
    )

def build_prompt_context(example):
    """
    Builds a consistent input prompt for the baseline.
    """
    return (
        f"### Context:\n{example['context']}\n\n"
        f"### {example['speaker']} says:\n{example['utterance']}\n\n"
        f"### Respond as {example['response_speaker']} the last utterance only.\n\n"
        f"{example['response_speaker']}:\n"
    )
