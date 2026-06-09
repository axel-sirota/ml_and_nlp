# Refactor Index — ML → NLP course

**Status legend**

| Tag | Meaning |
|-----|---------|
| 🆕 **NEW** | Notebook does not exist yet — build from scratch |
| ♻️ **RENEWED** | Heavy rebuild on old bones — most content rewritten |
| ✋ **TOUCHED** | Kept mostly as-is — light edits / move / renumber only |
| ☠️ **DIE** | Deleted outright |
| 📦 **ARCHIVE** | Moved to `Basics/` — out of core flow, kept for reference / pre-work |

---

## New course structure & status

| New # | New file | Status | From (old) |
|------:|----------|--------|------------|
| — | `Basics/Classical_Text_Classification.ipynb` | 📦 ARCHIVE | `1-Pre-NLP/2-Text_Classification_Classical.ipynb` |
| — | `Basics/PyTorch_Tensor_Exercises.ipynb` | 📦 ARCHIVE | `Frameworks/PyTorch_Tensor_Exercises.ipynb` |
| — | `Basics/PyTorch_Exercises.ipynb` | 📦 ARCHIVE | `Frameworks/PyTorch_Exercises.ipynb` |
| — | `Basics/Scikit_Learn_Exercises.ipynb` | 📦 ARCHIVE | `Frameworks/Scikit_Learn_Exercises.ipynb` |
| **A1** | `A-What-NLP-Can-Do/1-NLP_From_Scratch.ipynb` | ♻️ RENEWED | `1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb` |
| **A2** | `A-What-NLP-Can-Do/2-Pipeline_Tour.ipynb` | 🆕 NEW | — |
| **A3** | `A-What-NLP-Can-Do/3-Capstone_A.ipynb` | ♻️ RENEWED | `1-Pre-NLP/4-Capstone_1.ipynb` |
| **B4** | `B-How-It-Works/4-PyTorch_Tensors.ipynb` | ✋ TOUCHED | `Frameworks/PyTorch_Tensor_Exercises.ipynb` |
| **B5** | `B-How-It-Works/5-Word2Vec_and_Sentence_Embeddings.ipynb` | ♻️ RENEWED | `2-Text-Similarity/5-CBOW` + `7-Sentence_Transformers` |
| **B6** | `B-How-It-Works/6-PyTorch_NN_Basics.ipynb` | ✋ TOUCHED | `Frameworks/PyTorch_Exercises.ipynb` |
| **B7** | `B-How-It-Works/7-MLP_Word2Vec.ipynb` ★STOPPER | ✋ TOUCHED | `3-Text-Classification/8-MLP_Text_Classification.ipynb` |
| **B8** | `B-How-It-Works/8-Capstone_B.ipynb` | ♻️ RENEWED | `3-Text-Classification/10-Capstone_2.ipynb` |
| **C9** | `C-Transfer-Learning/9-DistilBERT_Chatbot.ipynb` | ♻️ RENEWED | `3-Text-Classification/9-BERT_Text_Classification.ipynb` |
| **C10** | `C-Transfer-Learning/10-EncoderDecoder_QA_Chatbot.ipynb` | 🆕 NEW | — |
| **C11** | `C-Transfer-Learning/11-Capstone_C.ipynb` | ♻️ RENEWED | `4-Text-Generation/13-Capstone_3_Shakespeare.ipynb` |

---

## ☠️ DIE — deleted outright

| Old file | Why it dies |
|----------|-------------|
| `1-Pre-NLP/3-Text_Processing_and_Classification_Metrics.ipynb` | Logreg + metrics — audience has ML; redundant |
| `2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb` | CBOW from scratch — beautiful but a time-sink (gensim used instead in B5) |
| `2-Text-Similarity/6-Pretrained_Embeddings_GloVe.ipynb` | GloVe fine-tune — embeddings depth not needed in core arc |
| `4-Text-Generation/11-LSTM_Text_Generation.ipynb` | RNN/LSTM detour — "no one uses RNN before"; attention motive lives in C9 |
| `4-Text-Generation/12-GPT2_Text_Generation.ipynb` | Decoder-only generation — cut; chatbot work is the generation payoff |

---

## Per-notebook detail files

Each touched/new/renewed notebook has its own topic plan in `notebooks/`:

- [A1 — NLP From Scratch](notebooks/A1-nlp-from-scratch.md) ♻️
- [A2 — Pipeline Tour](notebooks/A2-pipeline-tour.md) 🆕
- [A3 — Capstone A](notebooks/A3-capstone-a.md) ♻️
- [B4 — PyTorch Tensors](notebooks/B4-pytorch-tensors.md) ✋
- [B5 — Word2Vec + Sentence Embeddings](notebooks/B5-word2vec-sentence-embeddings.md) ♻️
- [B6 — PyTorch NN Basics](notebooks/B6-pytorch-nn-basics.md) ✋
- [B7 — MLP on Word2Vec ★STOPPER](notebooks/B7-mlp-word2vec.md) ✋
- [B8 — Capstone B](notebooks/B8-capstone-b.md) ♻️
- [C9 — DistilBERT Chatbot](notebooks/C9-distilbert-chatbot.md) ♻️
- [C10 — Encoder-Decoder Q&A Chatbot](notebooks/C10-encoder-decoder-qa-chatbot.md) 🆕
- [C11 — Capstone C](notebooks/C11-capstone-c.md) ♻️

See [plan.md](plan.md) for the overall refactor strategy, rationale, and execution order.
