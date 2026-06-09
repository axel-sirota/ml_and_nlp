# Core Technologies and Decisions - ML & NLP Course

Course manifest. Read by `/run-research-topic` and `/build-notebook` before any work.
Generic delivery (Data Trainers LLC). Scenarios stay client-neutral.

## Audience and goal

- **Audience**: practitioners with ML skills, weak in practice on PyTorch/TF/basic-ML, being
  brought into real-world NLP usage.
- **Goal / through-line**: every notebook advances toward a final deliverable - a fine-tuned
  transformer loaded into a simple **Gradio chatbot**.
- **Pain being fixed**: the old course front-loaded framework + from-scratch builds and never
  reached the word2vec -> MLP pivot. New arc reaches it fast and ends on transfer learning.

## The arc (see plans/refactor/plan.md + index.md)

- **Part A - What NLP can do**: tools-first NLP, then `pipeline()` across tasks, capstone.
- **Part B - How it works / build it**: PyTorch tensors -> w2v + sentence embeddings ->
  PyTorch nn -> MLP on word2vec (THE STOPPER) -> capstone.
- **Part C - Transfer learning**: DistilBERT + chatbot -> encoder-decoder Q&A + chatbot -> capstone.

## Stack

- **Framework**: PyTorch + HuggingFace (`transformers`, `datasets`, `evaluate`, `accelerate`,
  `sentence-transformers`), `gensim`, `spaCy`. NO TensorFlow/Keras. NO RNN/LSTM in core.
- **Environment**: Google Colab (GPU optional). Standard setup: `torch.cuda.is_available()`
  device select, `torch.manual_seed(42)`.
- **Placeholders**: `None # YOUR CODE` (exercise); full code + explanation (solution).
- **No external LLM API keys.** Everything runs from the HuggingFace Hub.

## Models

- **Encoder-only**: `distilbert-base-uncased` (classification, C9).
- **Encoder-decoder / seq2seq**: `t5-small` or `facebook/bart-base` (Q&A, C10).
- **Embeddings**: gensim pretrained vectors (e.g. `glove-wiki-gigaword-100`) + a
  `sentence-transformers` model (e.g. `all-MiniLM-L6-v2`).

## Datasets (public only)

| Use | Dataset | Source / id |
|-----|---------|-------------|
| Q&A (C10) | SQuAD | HF `squad` |
| Sentence similarity | STS-B | HF `glue`, config `stsb` |
| Sentiment | SST-2 | HF `glue`, config `sst2` |
| Reviews / sentiment capstone | IMDB | HF `imdb` |
| Reviews (classical / classification) | Yelp | HF `yelp_review_full` (or course Dropbox `yelp.csv`) |
| Topics / NER / routing | BBC, TWCS, news | course Dropbox CSVs (see reference memory) |

Course Dropbox CSVs (kept): Yelp, BBC, SMS Spam, CNN Headlines (news.csv), Twitter TWCS,
Airbnb train/test, Churn, GloVe 6B 100d. URLs in the reference-datasets memory file.

## Chatbot

- Simple **Gradio** app loading the fine-tuned model. Used by C9 (DistilBERT classifier
  chatbot) and C10 (seq2seq Q&A chatbot). Guard the Gradio cell so a no-Gradio env still
  runs the model-inference cells. Add `gradio` to requirements when C9/C10 are built.

## Conventions

- Build via `/build-notebook <slug> colab`, 5 cells at a time with approval.
- Theory -> Demo -> Lab; never chain more than 3 theory cells without a code cell.
- Each in-class lab: standard + stretch + homework extension.
- `numpy<2` pinned in every install cell.
- Public datasets only; verify ids/URLs before finalizing.
- No `sed`; edit notebooks per the notebook edit protocol.
- Plans live under `plans/refactor/`; this manifest lives at `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`.

## Out of scope (cut)

- TensorFlow / Keras (removed).
- RNN / LSTM (cut from core; attention motivation lives in C9).
- Decoder-only generation / GPT-2 (cut; chatbot work is the generation payoff).
- From-scratch CBOW training (replaced by gensim pretrained in B5).
- External LLM API usage (no OpenAI/Anthropic keys; HF Hub only).
