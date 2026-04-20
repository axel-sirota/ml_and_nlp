# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML & NLP progressive course by Axel Sirota (Data Trainers LLC). The course teaches NLP concepts from classical ML through deep learning, structured as Jupyter notebooks with exercises and solutions.

**Framework: PyTorch** (migrating from legacy TensorFlow/Keras). All new and updated notebooks should use PyTorch. Existing notebooks may still contain TensorFlow/Keras code pending migration.

## Environment Setup

```bash
python -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

All pip commands must use `.venv/bin/python3 -m pip`. All Python execution must use `.venv/bin/python3`.

## Course Structure

Four progressive modules, each ending with a capstone:

| Module | Directory | Topics |
|--------|-----------|--------|
| 1 | `1-Pre-NLP/` | Topic modeling, NER, text classification, logistic regression, boosting |
| 2 | `2-Text-Similarity/` | CBOW, Doc2Vec, fine-tuning embeddings |
| 3 | `3-Text-Classification/` | MLP for text (CNN headlines dataset, 130K news titles) |
| 4 | `4-Text-Generation/` | LSTM text generation (rental descriptions) |

Additional:
- `Frameworks/` — Standalone exercises for scikit-learn, Keras, TensorFlow 2
- `Solutions/` — Complete solutions mirroring the exercise structure (gitignored)

## Notebook Conventions

- **Naming**: `[NUMBER]-[Topic_Name].ipynb` (e.g., `8-ML_and_NLP_MLP.ipynb`)
- **Capstones**: `[NUMBER]-Capstone-Topic-[N].ipynb` at the end of each module
- **Structure per topic**: Theory markdown -> Demo code -> Lab with `None # YOUR CODE` placeholders
- **Exercise notebooks** have starter code; **Solution notebooks** have completed implementations
- **Datasets**: Public only (CSV from Dropbox URLs, sklearn datasets, HuggingFace datasets). No local data files checked in.

## Key Libraries

- **PyTorch** — Primary deep learning framework (target state)
- **scikit-learn** — Classical ML (logistic regression, boosting, train/test splits)
- **gensim** — Word2Vec, Doc2Vec embeddings
- **spacy 3.4** — NLP preprocessing, NER, tokenization (with spacy-transformers)
- **transformers** — HuggingFace for fine-tuning
- **pandas, numpy** — Data handling
- **matplotlib, seaborn** — Visualization
- **textblob** — Text processing utilities

## Common Patterns in Notebooks

- Imports follow: visualization -> data -> model -> training order
- Hyperparameters (`embedding_dim`, `epochs`, `batch_size`) defined at cell top
- GPU check at setup: `torch.cuda.is_available()` (PyTorch) or `!nvidia-smi`
- Tokenization often via `TextBlob(x).words`
- Data loaded with `pd.read_csv()` from Dropbox URLs

## Building Notebooks

Use the `/build-notebook` slash command. It enforces incremental building (5 cells at a time with approval), storytelling-driven teaching, and the theory->demo->lab structure. See `.claude/commands/build-notebook.md` for full protocol.
