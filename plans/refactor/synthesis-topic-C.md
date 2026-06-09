# Synthesis - Topic C (Transfer Learning / the chatbot payoff)

Notebooks: C9 (RENEWED, DistilBERT sentiment classifier chatbot) and C10 (NEW, encoder-decoder
T5 Q&A chatbot). Part C is the course's end-game: it opens the transformer black box one last
time and ships the artifact the whole course promised - a fine-tuned transformer running inside
a simple Gradio chatbot. C9 does the encoder-only (label-out) half; C10 does the encoder-decoder
(text-out) half and closes the course before the C11 capstone.

Plain ASCII only throughout. Both notebooks land inside the 20-25 cell budget at exactly 24
cells (0-23).

## 1. Teaching methodology that emerged across Topic C

The Part A/B methodology carries forward unchanged and is now visibly mature:

- **STAR opener, same persona.** Every notebook opens on a STAR (Situation / Task / Action /
  Result) story that continues the SAME support-platform ML-practitioner persona from Parts A
  and B. C9 picks up the B8 thread directly: the hand-built averaged-word2vec MLP plateaued
  because mean pooling reads "the update is not good" the same as "the update is good"; the task
  is to beat that ceiling with a fine-tuned DistilBERT and ship it. C10 continues the SAME team
  one beat later: leadership liked the C9 ticket-tagger and asked "can it actually ANSWER the
  customer from our help docs?", which a classifier cannot do, motivating the encoder-decoder.
  The STAR Result is always a concrete, shippable artifact, not a metric in a vacuum.

- **Fixed theory -> demo -> lab cadence with the hard 3-markdown rule.** Each concept runs
  theory markdown, then a fully-worked correct demo, then a lab. The "no more than 3 markdown
  cells chained without code" rule is enforced by explicit cell merges: C9 merges the attention
  intuition + DistilBERT-at-a-glance theory into one cell (6) so code (cell 7) follows
  immediately; C10 runs four named Concepts (1 three-shapes, 2 seq2seq tokenization, 3
  Seq2SeqTrainer, 4 generation), each theory cell backed quickly by a demo or lab.

- **Three-tier labs.** Every notebook ships (a) core verified labs with `None # YOUR CODE`
  placeholders and a PROVIDED verification block that prints/asserts, (b) exactly one labelled
  in-notebook STRETCH for fast finishers, and (c) one harder async HOMEWORK. C9: Lab 1
  (tokenization analysis, core) -> stretch "freeze the backbone, train only the head, measure
  the accuracy you gave up" -> homework "confidence-gate / selective-prediction abstention".
  C10: three core labs (preprocess fn, training args, answer_question fn) -> Stretch A (decoding
  -parameter sweep) + Stretch B (swap the T5 prefix to `summarize:`) -> 3-part homework (EM/F1
  via evaluate, hallucination probe, RAG grounding via B5 semantic_search).

- **The no-leak hint discipline (the rule that caused every prior defect).** Lab starter hints
  reference a helper, a role, or an output shape, and NEVER name the method/attribute/keyword/
  index/literal that solves the line. This was hardened in both plans: C9's Lab 1 was rewritten
  so hints never name the field, method, or pad id; C10's Defect 2 fix stripped the leaked
  target values (batch 16, LR 3e-4, 1 epoch, predict_with_generate=True) out of the Lab 2
  starter so they are role-only and point back to the theory cell. Verification blocks may
  still assert on properties (e.g. C10 asserts `learning_rate > 1e-4`) but the starter comments
  stay clean.

- **Provenance + research discipline.** Every cell is tagged FROM-OLD (with the old cell id) or
  FROM-NEW. RENEWED C9 carries a full Cell Migration Map (56 old cells -> 24) and a cited
  RESEARCH VALIDATED block; NEW C10 is all FROM-NEW with its own RESEARCH VALIDATED block. Cell
  budget is held by documented merges and deliberate drops (C9 drops the manual training loop +
  linear scheduler because B6/B7 already taught them; C10 fixed Defect 1 by merging two
  env-setup markdowns down to 24 cells).

## 2. How Topic C advances the chatbot through-line

This is where the long-promised artifact finally ships, and the architecture arc closes:

- **C9 - first real chatbot, encoder-only.** Parts A->B built the stack (tools-first NLP ->
  embeddings as geometry -> MLP on word2vec features). C9 opens the black box: attention
  replaces averaging, WordPiece shares subword roots, and a fine-tuned `distilbert-base-uncased`
  becomes a sentiment classifier that beats the B7/B8 averaged-embedding MLP ceiling (~90% on
  SST-2 validation vs the MLP's low-to-mid 80s). It is wrapped in a guarded `gr.Interface`: a
  teammate types a support message, the model returns POSITIVE/NEGATIVE with a confidence score.
  The save_pretrained -> reload -> pipeline -> Gradio pattern is established here.

- **C10 - generative finale, encoder-decoder.** The one-line C9->C10 bridge ("classification
  gives a label; a real assistant WRITES an answer") is paid off: students load `t5-small`, see
  it answer SQuAD questions poorly zero-shot (the deliberate "before"), fine-tune with
  `Seq2SeqTrainer`, generate with `.generate()` for a side-by-side "after", and drop the model
  into the SAME guarded Gradio shell - now with two text inputs (question, context) and a
  written-answer output. The closing reflection draws the explicit encoder-only (label) vs
  encoder-decoder (text) decision rule and shows that feeding C10's QA model a context retrieved
  by B5's `semantic_search` IS a RAG assistant.

- **Arc summary:** B5 made text into geometry and shipped RAG-style semantic search; B6/B7 built
  the MLP and the from-scratch training loop; C9 replaces averaging with attention and ships the
  first fine-tuned-transformer chatbot (classify); C10 ships the generative chatbot (answer) and
  names the next step. The Gradio guard pattern (try/except ImportError, share=True on Colab) and
  the save/reload pattern are intentionally reused verbatim across C9, C10, and forward into C11.

## 3. CONTINUITY - what the NEXT topic (C11 capstone) MUST reuse

Model ids (verbatim):
- `distilbert-base-uncased` (C9 fine-tune backbone, NUM_LABELS=2).
- `t5-small` (C10 seq2seq backbone, AutoModelForSeq2SeqLM).
- B5/A1 `all-MiniLM-L6-v2` (SentenceTransformer, 384-d) reused by C10's RAG homework via
  `embedder` + `semantic_search(query, top_k=3)` over `corpus`. C11 RAG path reuses the same.

Datasets:
- C9: `load_dataset("glue", "sst2")`; fields `sentence` / `label` / `idx`; 67349 train / 872
  validation / 1821 test. CRITICAL gotcha: GLUE test labels are hidden (-1), so EVALUATE ON THE
  VALIDATION SPLIT. Subsample train (`TRAIN_SUBSET = 6000`) for class time.
- C10: `load_dataset("squad")`; per example `question`, `context`, `answers` (a dict with
  `text` list + `answer_start` list). Take the FIRST gold answer: `answers["text"][0]`. Class
  subsets: `small_train = 2000`, `small_val = 200`.

Variable / helper names a follow-on notebook should match for coherence:
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` (torch.device object;
  convert to int 0/-1 ONLY for HF `pipeline(device=...)`). `SEED = 42` with `random.seed`,
  `np.random.seed`, `torch.manual_seed` (C9 uses full SEED block; C10 uses
  `torch.manual_seed(42)` + `np.random.seed(42)`).
- C9: `tokenizer`, `model`, `MODEL_NAME="distilbert-base-uncased"`, `NUM_LABELS=2`,
  `MAX_LENGTH=64`, `LR=2e-5`, `NUM_EPOCHS=2`, `BATCH_SIZE=32`, `id2label={0:"NEGATIVE",
  1:"POSITIVE"}` / `label2id`, `tokenize_fn`, `data_collator` (DataCollatorWithPadding),
  `compute_metrics`, `trainer` (Trainer), `SAVE_DIR="./distilbert_sst2_final"`, `chat_pipe`,
  `classify_message(text)`.
- C10: `build_input(question, context)` (returns `"question: {q}  context: {c}"`),
  `preprocess(examples)`, `max_input=256`, `max_target=32`, `tokenized_train`/`tokenized_val`,
  `data_collator` (DataCollatorForSeq2Seq), `training_args` (Seq2SeqTrainingArguments),
  `trainer` (Seq2SeqTrainer), `answer_question(question, context)`,
  `save_dir="t5-squad-qa-final"`, `chat_fn(question, context)`.

Scenario / pedagogy decisions to keep:
- Same support-platform persona; the running bar is "beat the prior model" (B7 LogisticRegression
  -> B8 MLP -> C9 transformer).
- Deliverable shape is ALWAYS a guarded `gr.Interface` (try/except ImportError; `share=True` for
  the Colab public link; non-Gradio fallback prints a direct inference call).
- save_pretrained writes BOTH model and tokenizer; reload BOTH; id2label travels in config.json
  so reloaded labels come back as words.
- C11 capstone explicitly tells students to pick ONE path: the C9 encoder-only classification
  chatbot OR the C10 encoder-decoder Q&A chatbot. C11 must support both shells.

API / pin facts to carry verbatim (do NOT regress):
- `transformers==4.57.1` (NEVER resolve to 5.x), `datasets>=2.19,<3`, `numpy<2` (course-wide),
  add `gradio`, `evaluate`, `accelerate`; `scipy<1.13` + `gensim==4.3.3` only when gensim is
  present (C10 deliberately does NOT install gensim/sentence-transformers).
- Colab restart-runtime note is REQUIRED after the install cell because numpy is downgraded.
- TrainingArguments uses `eval_strategy` (NOT the removed `evaluation_strategy`).
- C9 Trainer uses `processing_class=tokenizer` (the deprecated `tokenizer=` is gone). NOTE: C10's
  Seq2SeqTrainer still passes `tokenizer=tokenizer` in the current plan - see Gaps below.
- C9 rename gotcha: `rename_column("label", "labels")` or the model trains on nothing.
- C10 seq2seq specifics: T5 needs the task prefix; labels via `text_target=`; never build
  `decoder_input_ids` (teacher forcing is automatic); `DataCollatorForSeq2Seq` pads labels with
  -100; `predict_with_generate=True`; T5 LR ~3e-4 (higher than classification 5e-5); `fp16` only
  when `device.type == "cuda"`.

## 4. Gaps / risks for the next topic

- **C10 Trainer keyword inconsistency.** C9 standardized on `processing_class=tokenizer` and
  documents that `tokenizer=` is deprecated, but C10's `Seq2SeqTrainer(...)` (cell 16) still uses
  `tokenizer=tokenizer`. On transformers 4.57.1 this raises a deprecation warning, not an error,
  but C11 should standardize on `processing_class=` for both Trainer and Seq2SeqTrainer, or at
  minimum call out the inconsistency so students are not confused.
- **C10 `model` reassignment after save/reload (cell 20).** C10 rebinds the global `model` (and
  `tokenizer`) to the reloaded objects so `answer_question` uses the on-disk model. Any C11 code
  that reuses these globals after a reload must remember they now point at the reloaded instances,
  not the freshly trained `trainer.model`.
- **doc_vector / embedding-dim ambiguity inherited from B5.** B5's homework saves a 100-d `X`
  while the through-line narrates 384-d `all-MiniLM-L6-v2`. Topic C does not resolve this (it uses
  the 384-d `embedder` only for the C10 RAG homework, not as a classifier feature), so if C11's
  RAG path mixes B5 features it MUST pick one embedding width explicitly.
- **Generative hallucination is a known, deliberate teaching point, not a bug.** C10's homework
  has students feed an out-of-context question and watch T5 confidently invent an answer; the
  mitigation taught is RAG grounding. C11 must keep this honesty framing (and the "SQuAD-trained
  T5 mostly COPIES spans" caveat) rather than presenting the QA chatbot as reliable open-domain.
- **GPU dependence and run times.** Both fine-tunes target a Colab T4 (~3-6 min each); on CPU
  they are much slower. C11 should keep the same subset sizes / fp16-on-GPU-only guard so the
  capstone still fits a class session.
- **No accuracy/EM assertion gate in-notebook.** C9 narrates ~0.90 and C10 shows a qualitative
  before/after; neither hard-asserts a metric floor (C10 defers EM/F1 to homework). If C11 needs
  a pass/fail gate, it must add the metric computation itself.
- **share=True external dependency.** The Gradio public-link launch depends on Gradio's tunneling
  service being reachable; the try/except guards ImportError but not a network failure at
  `.launch()`. Acceptable for class, but worth knowing for an offline run.
