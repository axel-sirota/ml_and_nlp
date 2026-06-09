# Refactor Plan — ML → NLP course

**Status:** APPROVED blueprint, NOT yet executed (notebooks untouched).
**Date:** 2026-06-08
**Companion:** [index.md](index.md) (status of every notebook) · `notebooks/*.md` (per-NB topics)

---

## 1. Goal & the problem we fix

**Audience:** people with ML skills, weak in practice on PyTorch/TF/ML-basics, being
brought into **real-world NLP usage**.

**The pain:** the old course front-loads framework + ML-basics + from-scratch builds,
so class **never reaches the word2vec→MLP notebook** (the pivot).

**Design principles:**
1. Tools-first — show what NLP *does* before how it works.
2. `pipeline()` early — transformers in 3 lines (the WOW + motivation).
3. Keep the word2vec → simple-NN bridge ("embeddings as features").
4. Cut the RNN/LSTM detour — attention motivation already lives in the BERT notebook.
5. Finish on transfer learning: encoder-only → encoder-decoder, each loaded into a chatbot.
6. Framework / from-scratch material → `Basics/` (archive / pre-work), not core flow.

---

## 2. The arc

```
PART A  "What NLP can do"      A1 from-scratch → A2 pipeline tour → Capstone A
   │  "ok... now let's build it"
PART B  "How it works"         B4 tensors → B5 w2v+sent-emb → B6 nn → B7 MLP★ → Capstone B
   │  "now use the big pretrained ones"
PART C  Transfer learning      C9 DistilBERT+chatbot → C10 enc-dec Q&A+chatbot → Capstone C
```

★ B7 (MLP on word2vec) = THE STOPPER — the notebook class must reliably reach.
🤖 Chatbot = simple **Gradio** app loading the fine-tuned model (C9 + C10).

---

## 3. Folder structure (Full rename A/B/C + Basics/)

```
Basics/                          archive / optional / pre-work
A-What-NLP-Can-Do/               Part A
B-How-It-Works/                  Part B
C-Transfer-Learning/             Part C
```

Full old→new mapping and status tags: see **[index.md](index.md)**.

---

## 4. Execution order

**Phase 0 — structure (safe, mechanical, do first):**
1. Create folders: `Basics/ A-What-NLP-Can-Do/ B-How-It-Works/ C-Transfer-Learning/`.
2. `git rm` the 5 DIE notebooks (old NB3, NB5, NB6, NB11, NB12).
3. `git mv` keeps/archives into new homes (Classical→Basics, Frameworks→Basics,
   old NB8→B7, old NB9→C9, capstones, etc.) with renumbering.
4. Update `CLAUDE.md`: new module table, CORE-vs-Basics legend, audience/prereq line.
5. Update `requirements.txt`: add `gradio` (chatbot dep).

**Phase 1 — new content (via /build-notebook, 5 cells at a time, approval each step):**
6. A2 Pipeline Tour 🆕
7. C10 Encoder-Decoder Q&A Chatbot 🆕

**Phase 2 — renews & touches:**
8. A1 rebuild · A3 capstone rework
9. B5 merge (gensim w2v + sentence-transformers) · B4/B6 trim from Frameworks · B8 capstone
10. C9 add Gradio chatbot · C11 capstone rework

**Phase 3 — finalize:**
11. Regenerate `Solutions/` mirror (gitignored) to match new structure.
12. Run `validate_notebooks.py` on every exercise/solution pair.
13. Update slides / course-map references.

---

## 5. Constraints & rules (from CLAUDE.md)

- PyTorch + HuggingFace only. No TF/Keras. No RNN/LSTM in core.
- All python via `.venv/bin/python3`; pip via `.venv/bin/python3 -m pip`.
- New deps → add to `requirements.txt`.
- Build notebooks with `/build-notebook` (5 cells, approval checkpoints).
- Datasets: public only (Dropbox CSV / sklearn / HuggingFace datasets).
- No `sed`; edit one-by-one.

---

## 6. Open / deferred decisions

- **C10 dataset:** SQuAD vs custom Q&A — decide at build time.
- **Decoder-only generation (GPT-2):** CUT. Returns only as optional `Basics/` if wanted later.
- **Gradio in Colab:** confirm `share=True` works in the course's runtime.
- **Capstone scope:** each Part ends with a capstone (A/B/C) reworked to the new arc.

---

## 7. Risk notes

- Renaming folders breaks any hard-coded paths in slides / READMEs → grep & fix in Phase 0.4.
- B7 (stopper) is TOUCHED not rebuilt — protect its content; it's the whole point.
- Gradio adds a heavier dep; keep the chatbot cell optional/guarded so a no-Gradio
  environment still runs the model inference cells.
