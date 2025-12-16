# NPAIC: Knowledge-Grounded NPC Dialogue Generation with Personality and Memory

An experimental system for generating dynamic, character-consistent NPC dialogue in role-playing games using large language models, memory mechanisms, and parameter-efficient fine-tuning.

## Overview

NPAIC (NPC AI Character) explores whether lightweight LLMs augmented with explicit memory and character-specific adaptation can generate immersive dialogue that remains faithful to established game characters. Using Arthur Morgan from Red Dead Redemption 2 as a case study, this project evaluates different approaches to NPC dialogue generation.

### Key Features

- **Memory-Augmented Generation**: Conversation history summarization and knowledge graph integration
- **Character-Specific Adaptation**: LoRA-based parameter-efficient fine-tuning
- **Real-Time Inference**: Sub-second response latency on consumer hardware
- **Modular Architecture**: Swappable character weights and memory systems

## Architecture

```
Player Input → Dialogue Context → Memory Modules → LLM → NPC Response
                                  ├─ Conversation Memory (summarization)
                                  └─ Knowledge Graph Memory (entity retrieval)
```

The system integrates three core components:

1. **Base Language Model**: Qwen2.5-3B for real-time dialogue generation
2. **Explicit Memory Mechanisms**: Summarized conversation history and retrieved world knowledge
3. **Character Adaptation**: LoRA adapters trained on character-specific dialogue

## Installation

```bash
# Clone the repository
git clone https://github.com/gannonsmith/npaic.git
cd npaic

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- NetworkX
- sentence-transformers

## Dataset

The project uses fan-transcribed dialogue from Red Dead Redemption 2, focusing on Arthur Morgan:

- **Training Set**: 2,528 dialogue pairs
- **Validation Set**: 316 pairs
- **Test Set**: 316 pairs

Each instance includes:
- Mission identifier
- Prior dialogue context
- Current speaker utterance
- Target NPC response

## Evaluation Results

| Model | Character Consistency | Relevance & Coherence | Task Alignment | BLEU | BERTScore |
|-------|----------------------|---------------------|----------------|------|-----------|
| Random | 1.93 | 1.62 | 0.15 | 0.003 | 0.845 |
| In-Character Random | 2.01 | 1.79 | 0.11 | 0.003 | 0.846 |
| Embedding Retrieval | 3.78 | 3.14 | 0.43 | 0.028 | 0.854 |
| **Base Model** | **3.19** | **4.07** | **0.51** | 0.004 | 0.842 |
| LoRA + Memory | 2.75 | 3.17 | 0.40 | 0.003 | 0.825 |
| Actual Response | 4.90 | 4.82 | 0.84 | 0.849 | 1.000 |

### Key Findings

- **Base model outperforms fine-tuned variant**: Raw dialogue context proves more effective than summarized memory
- **Retrieval preserves character voice**: Embedding-based retrieval excels at stylistic authenticity but lacks flexibility
- **Memory summarization degrades performance**: Information loss during summarization harms coherence and alignment

## Project Structure

```
npaic/
├── data/               # Dialogue datasets
├── models/             # Stored models and LoRA weights
├── results/            # Evaluation outputs and metrics
└── src/                # Core source code
    ├── evaluation/         # Metrics and LLM-based judges
    ├── generation/         # Dialogue generation logic
    ├── knowledge/          # Character-specific knowledge graphs
    ├── memory/             # Conversational memory management
    ├── personality/        # Personality training
    ├── preprocessing/      # Preprocessing
    └── util/               # Utility functions
```

## Limitations

- **Not production-ready**: Output formatting inconsistencies and occasional verbosity
- **Limited training data**: Only 2,528 examples for character adaptation
- **Memory loss**: Summarization can lose conversational nuance
- **Single character focus**: Designed for Arthur Morgan specifically

## Future Work

1. **Systematic ablation studies** to isolate memory component contributions
2. **Advanced summarization methods** using RL optimization
3. **Automated prompt tuning** through gradient-based or evolutionary approaches
4. **Expanded knowledge graphs** with richer character lore and relationships
5. **Multi-character support** to generalize across different NPCs and games
6. **LoRA hyperparameter optimization** for better stability and performance

## Related Work

This project builds on:
- **LaMP** (Salemi et al., 2024): Personalized text generation benchmarks
- **Knowledge-grounded dialogue** (Ashby et al., 2023): Structured knowledge integration
- **Generative agents** (Park et al., 2023): Memory architectures for persistent characters
- **LoRA** (Hu et al., 2021): Parameter-efficient fine-tuning
- **LLM-as-judge** (Li et al., 2024): Automated evaluation methods

## Citation

```bibtex
@misc{smith2025npaic,
  author = {Smith, Gannon},
  title = {NPAIC: Knowledge-Grounded NPC Dialogue Generation with Personality and Memory},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gannonsmith/npaic}}
}
```

## License

This project is for academic and research purposes. Red Dead Redemption 2 dialogue data is used under fair use for non-commercial research.

## Acknowledgments

Thanks to the CSE 595 course instructors and classmates for feedback throughout the project. All work is original research by Gannon Smith at the University of Michigan.

---

**Note**: This is an experimental research project exploring the feasibility of LLM-based NPC dialogue generation. It demonstrates promising directions but highlights significant challenges in memory design and character consistency that require further research.