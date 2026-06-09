# Synthesis - Topic A: What NLP Can Do

Scope: the three notebooks that open the ML and NLP course.

- A1 - `A-What-NLP-Can-Do/1-NLP_From_Scratch.ipynb` (RENEWED, 23 cells)
- A2 - `A-What-NLP-Can-Do/2-Pipeline_Tour.ipynb` (NEW, 25 cells)
- A3 - `A-What-NLP-Can-Do/3-Capstone_A.ipynb` (RENEWED, 24 cells)

This document captures the teaching methodology that emerged across Topic A, how the
notebooks advance the course's chatbot through-line, the concrete artifacts the next
topic (Part B) must reuse, and the open risks.

---

## 1. Teaching Methodology

### STAR framing (one running scenario, escalating)
Every notebook opens on a STAR story and every notebook keeps the SAME persona: you are
the ML / NLP person dropped onto a real team with raw text and no models. The scenario
escalates across the topic:

- A1 (Situation): a media-analytics team hands you a folder of unlabeled text. Two
  questions: "who/what is mentioned?" and "what is it about?" Answered with the classical
  toolbox, zero training.
- A2 (Situation): a customer-support platform needs sentiment, routing, entities, and
  summaries shipped this sprint. Answered with `pipeline()`, zero training, six tasks.
- A3 (Situation): the SAME support platform now needs a real ticket router on a real noisy
  corpus (TWCS). Answered with zero-shot classification + NER, zero training, plus a
  confidence fallback and a Gradio app.

The "no model training in Part A" promise is stated explicitly and repeatedly. Part A is
deliberately a tools-USING topic; training is held back as the motivation for Part B/C.

### Theory -> Demo -> Lab cadence
Each section follows a fixed rhythm: a short markdown theory cell, a heavily-commented
runnable demo cell, then a hands-on lab. The hard rule observed across all three plans is
"no more than 3 theory/markdown cells without a code cell." Demos are fully worked and
correct; labs hand the student a starter with `None  # YOUR CODE` placeholders that NEVER
leak the answer (the "cover-the-solution" test). Every lab starter ends with a provided
verification block that asserts shape/correctness without revealing the solution.

### Lab tiers (consistent three-tier structure)
Each lab is built in tiers so fast and slow students both have a path:
1. Core task (the must-do, verified).
2. Stretch (labelled, in-notebook, e.g. entity-surname normalization in A1, a confidence
   guardrail in A2, choosing k via silhouette in A1).
3. Homework extension (async, genuinely harder, e.g. ORG/GPE co-occurrence graph in A1,
   BERTopic comparison, a confidence-thresholded router in A2, multi-label routing and
   "ship it to a HuggingFace Space" in A3).

### Provenance discipline
Every cell carries a `Provenance:` line (FROM-OLD <old notebook> <cell id>, or FROM-NEW).
A1 = 21 FROM-OLD / 2 FROM-NEW, A2 = 0/25 (net-new), A3 = 15 FROM-OLD / 9 FROM-NEW. Each
RENEWED plan ships a cell migration map and a RESEARCH VALIDATED block with cited sources.
Output is plain ASCII (no em/en dashes, no Unicode multiplication sign, no emoji).

### Cell budget
All three notebooks land in the 20-25 cell ceiling (A1=23, A2=25, A3=24). Plans show
explicit cell-merge surgery to stay inside the budget.

---

## 2. Chatbot Through-Line (the narrative arc)

The course end goal is a fine-tuned transformer served in a simple Gradio chatbot. Topic A
plants every seed of that goal without training anything:

- A1 establishes the VOCABULARY (tokens, lemmas, entities, topics) and introduces the
  text-to-vector idea via the `all-MiniLM-L6-v2` sentence embedder. This is the explicit
  "text -> vector" seed reused as MLP features in Part B and as the DistilBERT backbone idea
  in Part C. A1 closes by motivating A2: "all this took installs, stopwords, casing care,
  and clustering plumbing - watch one line of pipeline() do the same."
- A2 collapses every A1 task into one line of `pipeline()` across six tasks. The crucial
  through-line move: the DEFAULT sentiment pipeline is
  `distilbert-base-uncased-finetuned-sst-2-english`, i.e. exactly `distilbert-base-uncased`
  (the Part C fine-tune target) already fine-tuned on SST-2 (a Part C dataset). Students use
  the finished product before building it. A2 also previews `gr.Interface.from_pipeline` so
  students see a pipeline IS already a chatbot backend. A2 closes by motivating Part B:
  "the pipeline is a sealed black box; in Part B we build every piece (tensors, embeddings,
  neural nets, MLP), in Part C we fine-tune distilbert and ship it in Gradio."
- A3 makes the case for fine-tuning VISCERAL. Students ship a working zero-shot router on
  TWCS, measure that it tops out around 70-85 percent, and end on a guarded `gr.Interface`
  mini-app - the EXACT Gradio UX skeleton the Part C chatbot reuses, just a different model
  behind it. A3's one-line bridge to B4: "zero-shot got us 70-something percent with no
  training; to beat it we need tensors, embeddings, and training loops, then fine-tune a
  real transformer. That is Part B."

Arc in one sentence: A1 names the concepts and meets the embedder -> A2 shows the one-line
pretrained shortcut and the fine-tune target (distilbert/SST-2) plus the Gradio shape ->
A3 ships a real zero-shot app, hits the accuracy ceiling, and hands off to Part B to open
the black box and earn the fine-tune.

---

## 3. CONTINUITY - what Part B (and C) MUST reuse

### Models / model ids (carry these forward verbatim)
- `all-MiniLM-L6-v2` (sentence-transformers, 384-dim, distilled 6-layer BERT, CPU-friendly):
  introduced in A1 as the "text -> vector" through-line artifact. Part B should reuse this
  exact embedder when it needs document/sentence features for the MLP.
- `distilbert-base-uncased`: named in A2 (and A3 wrap-up) as the Part C fine-tune target. Its
  fine-tuned form `distilbert-base-uncased-finetuned-sst-2-english` is the default sentiment
  pipeline. SST-2 is the named dataset. Part C must honor this exact pairing.
- `facebook/bart-large-mnli`: the zero-shot default used in A2 and A3.
  `MoritzLaurer/deberta-v3-base-mnli-fever-anli` is the named stronger backbone (A3 homework).
- `distilbert-base-cased-distilled-squad` (QA, A2; SQuAD dataset named for the Part C Q&A
  chatbot). `sshleifer/distilbart-cnn-12-6` (summarization). `distilroberta-base` (fill-mask).
  `cardiffnlp/twitter-roberta-base-sentiment-latest` (the NEUTRAL-aware swap model in A2).

### Datasets
- A1 corpus: `load_dataset("SetFit/bbc-news", split="train").to_pandas()`. CRITICAL DECISION:
  use the RAW HuggingFace dataset, NOT the pre-stemmed Dropbox `bbc.csv`. Columns are
  `text` (raw, naturally-cased article body) and `label_text`
  (business/entertainment/politics/sport/tech). The pre-processed CSV is explicitly REJECTED
  because lowercased+stemmed text breaks NER, displacy, the PERSON lab, lemmatization, and the
  casing experiment. Any later notebook reusing BBC must use the HF dataset, not the CSV.
- A3 corpus: TWCS (Twitter Customer Support), loaded via `pd.read_csv(TWCS_URL, nrows=...)`
  from the course Dropbox URL with `dl=1`
  (`https://www.dropbox.com/scl/fi/6w2dchvsthwdol934nprt/twcs.csv?rlkey=u5205iehhrog88qt8iwm4udxs&dl=1`).
  Schema: tweet_id, author_id, inbound, created_at, text, response_tweet_id,
  in_response_to_tweet_id. No routing column; weak labels recovered from reply metadata.
- Datasets named for Part C: SST-2 (sentiment), SQuAD (QA).

### Helper functions and variable names (reuse exactly to keep the course coherent)
- A1: `nlp = spacy.load("en_core_web_sm")`, `clean_text(text)`, `preprocess(text)`,
  `embedder = SentenceTransformer("all-MiniLM-L6-v2")`, `embedder.encode(...)`,
  `KMeans(n_clusters=k, n_init=10, random_state=42)`. Dataframe columns: `bbc["clean"]`,
  `bbc["cluster"]`; column refs `text_col="text"`, `cat_col="label_text"`.
- A2: pipeline objects `classifier`, `router`, `ner`, `qa`, `summarizer`, `unmasker`;
  `candidate_labels`, `device = 0 if torch.cuda.is_available() else -1`. NOTE the deliberate
  naming choice: A2 does NOT reuse A1's `nlp`/`doc` names to avoid collision.
- A3: `clf` (zero-shot pipe), `ner` (NER pipe), `reply_map`, `route_tweet(text, threshold)`,
  `extract_entities(text)` (defined once in Lab 4, REUSED by `route_tweet` - not re-inlined),
  `CANDIDATE_LABELS`, `HYPOTHESIS_TEMPLATE`, `CONF_THRESHOLD = 0.5`, columns
  `labeled`, `routing_set['route']`, `routing_set['confidence']`, `routing_set['final_route']`,
  and the `"ESCALATE_TO_HUMAN"` fallback sentinel.

### Scenario / convention decisions Part B must honor
- The persona is the same "ML person on a team" thread; Part B should continue it (now: open
  the black box so we can fine-tune past the zero-shot ceiling).
- Device convention: pipelines use the int `device` arg (0 = first GPU, -1 = CPU). Part B,
  using PyTorch directly, will switch to `torch.device(...)` - flag this transition.
- Reproducibility convention: `SEED = 42`, `np.random.seed(SEED)`, `torch.manual_seed(42)`.
- Gradio convention: the final deliverable is `gr.Interface` / `gr.Interface.from_pipeline`,
  always guarded with `try/except ImportError`. Part C reuses this exact shell.
- numpy is pinned `<2` course-wide.

### Dependency pins (important and slightly inconsistent - see risks)
- A1: `spacy>=3.7,<3.9`, `sentence-transformers>=2.7`, `datasets>=2.19`, `numpy<2`. No
  bertopic, no gensim in core (gensim needs numpy<2 AND scipy<1.13, kept out on purpose).
- A2: `transformers==4.57.*` (PINNED to 4.x: transformers 5.x REMOVED the summarization,
  question-answering, and translation pipelines and requires numpy 2). `datasets>=2.19,<4`,
  `evaluate>=0.4.2`, `accelerate>=0.30`, `numpy<2`.
- A3: `transformers>=4.40`, `datasets>=2.19`, `accelerate>=0.30`, `numpy<2`.

---

## 4. Gaps / Risks for the next topic

1. A3 plan verify status is FAIL in the orchestration handoff (A1 and A2 PASS). The plan
   body claims both defects were fixed (cell count 24, and the Lab 4 reveal resolved by
   reusing `extract_entities` in `route_tweet`), but the FAIL flag means A3 should be
   re-verified before `/build-notebook` is run on it. Part B should not assume A3's notebook
   is final.

2. transformers version pin is INCONSISTENT across the topic. A2 hard-pins
   `transformers==4.57.*` (because 5.x dropped summarization/QA/translation pipelines and
   forces numpy 2). A3 uses the looser `transformers>=4.40`, which on a fresh Colab could
   resolve to 5.x and break the numpy<2 pin and any pipeline expectations. Part B and C
   should standardize on the 4.57.x pin (or explicitly justify a different one) for the whole
   course to avoid a numpy 2 / pipeline-removal collision.

3. numpy<2 is load-bearing course-wide. gensim (numpy<2 AND scipy<1.13) and bertopic are kept
   OUT of A1 core for this reason. If Part B reintroduces gensim (Word2Vec/Doc2Vec per the
   course CLAUDE.md module map), it must re-pin scipy<1.13 as well, or it will fail to import.

4. Device convention changes at the Part A -> B boundary. A1/A2/A3 use the pipeline int
   `device` (0 / -1). Part B moves to raw PyTorch and `torch.device("cuda"/"cpu")`. This is a
   real teaching seam to address explicitly so students are not confused by two device idioms.

5. Naming hygiene: A2 deliberately avoided reusing A1's `nlp`/`doc`. Part B should likewise
   avoid colliding with the carried-forward names above (`embedder`, `classifier`, `clf`,
   `ner`, `route_tweet`, `extract_entities`, `clean_text`, `preprocess`) unless intentionally
   reusing the same artifact.

6. The fine-tune promise is now load-bearing on specifics: A2/A3 explicitly promise Part C
   will fine-tune `distilbert-base-uncased` and serve it in Gradio, and name SST-2 and SQuAD.
   Part C MUST deliver exactly that model id and dataset pairing, or the through-line breaks.

7. A1's `all-MiniLM-L6-v2` is promised as the Part B MLP feature source ("you use the same
   text -> vector idea as features for a small neural net"). Part B must actually reuse this
   embedder (or explicitly explain a substitution) to honor the seed A1 planted.
