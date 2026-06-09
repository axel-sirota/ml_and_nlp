# Synthesis - Topic B (How It Works): B4 + B5

Scope: the first two notebooks of Part B ("open the black box").
- B4 -> `B-How-It-Works/4-PyTorch_Tensors.ipynb` (slug `pytorch-tensors`, 25 cells, TOUCHED, FROM-OLD 24 / FROM-NEW 1)
- B5 -> `B-How-It-Works/5-Word2Vec_and_Sentence_Embeddings.ipynb` (slug `word2vec-sentence-embeddings`, 24 cells, RENEWED, FROM-OLD 22 / FROM-NEW 2)

Note on continuity within Topic B: both plans repeatedly forward-reference B6 (nn.Module / loss
/ optimizer / DataLoader) and B7 (the MLP "stopper" trained on doc vectors). B6 and B7 are NOT
part of this synthesis pair but are the immediate next rungs; their contracts are spelled out in
the CONTINUITY section so they (and Part C) stay coherent.

---

## 1. Teaching methodology that emerged across Topic B

The Part A methodology carries forward unchanged and is honored cell-for-cell in both plans:

- STAR opener on the same escalating persona. Every notebook opens on a Situation/Task/Action/
  Result story that continues the one ML/NLP-practitioner-on-a-real-team thread. B4 picks the
  persona up directly at the Part A handoff ("you shipped a zero-shot tweet router that tops out
  around 70-85 percent; the team wants better, which means training, which means PyTorch"). B5
  keeps the same customer-support-platform team and gives them a concrete new ask (semantic search
  over the support knowledge base, plus reusable classifier features without a per-request GPU
  bill). The "no model training in Part A" promise that was held back through A1-A3 is now being
  paid off: B4's whole framing is "this is where we open the black box."

- Fixed theory -> demo -> lab cadence, hard "no more than 3 markdown cells without code" rule.
  Both notebooks are organized into short sections, each a Theory markdown, a fully-worked correct
  Demo, then a Lab. B4 runs five such cycles (Basics, Operations, Reshaping, Indexing+Broadcasting,
  Autograd). B5 runs three lab cycles across three parts (word vectors; sentence embeddings;
  semantic search). The verification checklists in both plans explicitly assert the cadence and the
  3-markdown-cell ceiling.

- Lab starters use `None  # YOUR CODE` placeholders that never leak the answer (the cover-the-
  solution test), and every lab ends in a provided verification block. This is the most heavily
  enforced rule in both plans and was the source of all three of B5's adversarial defects. The
  discipline that emerged: hints point to helpers and the RIGHT-shaped output by ROLE, never naming
  the method/attribute/index that solves the line. Concretely - B4 Lab 5 references "the task-2
  expression" instead of spelling out `p*q + p**2`; B5 Lab 2 says "you already used a cosine helper
  from util ... pull the scalar out" instead of writing `util.cos_sim(...).item()`, and Lab 3 says
  "the util search helper from the demo ... ask it for just the top hit" instead of naming
  `util.semantic_search(..., top_k=1)[0]`.

- Three-tier labs: core verified task, a labelled in-notebook stretch for fast finishers, and a
  genuinely harder async homework extension. B4 Lab 5 is the clearest example: core autograd tasks,
  an in-notebook "gradient descent by hand" stretch (so students see what `optimizer.step()` will
  automate), and a production-grade homework (finite-difference gradient checking, the idea behind
  `torch.autograd.gradcheck`). B5's homework is the load-bearing one for the through-line: it builds
  `doc_vector()` and saves `X, y` as the literal input to the B7 MLP.

- Provenance discipline and RENEWED extras. Every cell carries a FROM-OLD/FROM-NEW provenance line
  citing the exact source cell id. RENEWED plans additionally ship a Cell Migration Map and a cited
  RESEARCH VALIDATED block; B5 (RENEWED, merging two old notebooks) has both, B4 (TOUCHED, lifting 5
  of 8 sections from one source) has a migration map plus its own research block.

- Cell-budget surgery. Both notebooks stay inside the 20-25 cell ceiling (B4=25, B5=24) via
  explicit cell merges documented in the migration map - B4 collapses the old indexing and
  broadcasting sections into four cells (17-20) and folds autograd lab+stretch+homework into one
  composite Cell 23; B5 merges load+encode+compare into single cells and drops the entire from-
  scratch CBOW training pipeline.

- Plain-ASCII output: no em/en dashes, no Unicode multiply sign, no emoji. Asserted in both
  verification checklists.

---

## 2. How Topic B advances the chatbot through-line

End goal (unchanged): a fine-tuned transformer (DistilBERT) loaded into a simple Gradio chatbot.
Part A ended by naming the embedder, revealing the fine-tune target, and shipping the guarded
gr.Interface skeleton, then handed off to Part B "to open the black box and earn fine-tuning."
Topic B is the first half of opening that box, and the arc from B4 to B5 is tight:

- B4 lays the single most load-bearing brick under the whole goal: autograd. Its one-line story is
  "every text becomes a tensor, and `.backward()` is how a model learns from it." Cell 4 (the one
  FROM-NEW cell) draws the entire chain explicitly: today a tensor is a NumPy array plus GPU plus
  autograd; B5 turns each sentence into a 384-dim float vector (a 1-D tensor) from `all-MiniLM-L6-
  v2`; B6 flows those vectors through `nn.Linear` (matrix multiply plus a broadcasted bias add,
  both written by hand in B4); B7 trains an MLP on the 384-dim tensors by calling `.backward()` in
  a loop; C9 uses the same `.backward()` to fine-tune DistilBERT for the chatbot. So when the
  student runs gradient descent by hand at the end of B4, they are running the literal engine that
  trains everything that follows.

- B5 is the "text becomes geometry" rung. It opens the A1 black box one level: vectors live in a
  space, cosine similarity reads it, and you can search and cluster with nothing but geometry. Two
  concrete through-line payloads:
  (a) the semantic search engine (Cell 21, `semantic_search`) is explicitly framed as the retrieval
      primitive a RAG chatbot runs before it answers - encode the corpus offline, encode the query
      online, nearest-neighbor in vector space;
  (b) the homework `doc_vector()` (mean-pool of word2vec) is the exact feature matrix B7's MLP
      consumes - B5 literally hands B7 its inputs and the LogisticRegression baseline the MLP must
      beat.

- B5 also plants the motivation for Part C. Static word2vec has one vector per word, so "bank"
  (river) and "bank" (money), or "python" the language and the snake, collapse into a single blended
  vector. That polysemy gap is named on purpose (Cell 9 demo comment, Cell 10/17 caveats) as the
  explicit reason Part C reaches for a contextual transformer (DistilBERT). SBERT is flagged as
  paraphrase-aware but NOT polarity-aware, so sentiment still needs the Part C sentiment-tuned
  classifier.

Bridges are stated in-text: B4 ends "next we turn real sentences into the 384-dim float tensors
(from all-MiniLM-L6-v2) that this exact machinery will train on." B5 ends "text is now a fixed-size
vector; B6 builds the PyTorch neural-net machinery (nn.Module, loss, optimizer, DataLoader); B7
points it at these embeddings and beats the baseline you just measured."

---

## 3. CONTINUITY - what the NEXT topic (B6/B7, then Part C) MUST reuse verbatim

### Models / ids (reuse exactly; do not rename, do not swap)
- `all-MiniLM-L6-v2` - the sentence encoder, named `embedder` (a `SentenceTransformer`). Same id as
  A1. Output dimension is 384. THIS IS THE B7 MLP INPUT WIDTH and the through-line spine. B6/B7 must
  not substitute another encoder.
- `glove-wiki-gigaword-100` - pretrained word vectors loaded via `gensim.downloader.api.load(...)`
  into `wv` (a `KeyedVectors`): 400,000 words, 100 dimensions. The mean-pool `doc_vector()` feature
  is 100-d (NOT 384-d). Watch this distinction: B5 produces TWO different document representations
  (100-d averaged-word2vec for the homework baseline, 384-d SBERT for search). B7's MLP input width
  depends on which one is fed in - the B5 homework saves the 100-d `X` from `doc_vector`, so if B7
  intends to use the 384-d SBERT vectors instead it must re-encode and re-save. RESOLVE THIS
  EXPLICITLY in B7.
- `distilbert-base-uncased` (Part C fine-tune target) and its fine-tuned form
  `distilbert-base-uncased-finetuned-sst-2-english`, dataset SST-2. Unchanged from Part A carry-
  forward. Part C must deliver exactly this or the through-line breaks.
- `distilbert-base-cased-distilled-squad` (QA, dataset SQuAD) for the Part C Q&A chatbot path.
- Carried but not used in B4/B5 (keep available for Part C): `facebook/bart-large-mnli` (zero-shot)
  with stronger `MoritzLaurer/deberta-v3-base-mnli-fever-anli`; `sshleifer/distilbart-cnn-12-6`;
  `distilroberta-base`; `cardiffnlp/twitter-roberta-base-sentiment-latest`.

### Variable names / helpers introduced in Topic B (reuse, do not re-inline or redefine)
- B4: `SEED = 42`, `torch.manual_seed(SEED)`, `np.random.seed(SEED)`,
  `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. The device idiom is now
  a `torch.device` object (this is the A->B seam change away from Part A's pipeline int device of
  `0 if cuda else -1`). B6/B7/Part C training code uses the `torch.device` form; only HuggingFace
  `pipeline(device=...)` calls take the int form.
- B5: `wv` (KeyedVectors), `embedder` (SentenceTransformer), `analogy(a, b, c)` helper,
  `mean_vector(sentence)` helper, `cos(u, v)` helper, `semantic_search(query, top_k=3)` and
  `best_match(query)` (Lab 3), `corpus` / `corpus_emb` (the cached support-KB corpus). The homework
  introduces `doc_vector(text)` and saves `X` (shape (N, 100)) and `y`.
- B5 deliberately avoids name collision with B4: B4 owns tensor scratch names (`x`, `w`, `y`, `a`,
  `b`, `p`, `q`); B5 owns `wv`, `embedder`, `corpus`, `corpus_emb`. The next topic should likewise
  not reuse `x`/`w`/`y` for anything model-level if it also needs the B5 names live.

### Datasets and how they are loaded (reuse the exact loader call)
- STS-B (B5 Lab 2): `load_dataset("glue", "stsb", split="validation")`, fields `sentence1`,
  `sentence2`, `label` (0-5 float gold). Evaluated with Spearman rank correlation (`scipy.stats.
  spearmanr`), target ~0.8.
- SST-2 (B5 homework feature-build AND Part C fine-tune): `load_dataset("glue", "sst2", ...)`,
  fields `sentence` and `label`. B5 homework uses ~500 train rows to build `X, y`; Part C fine-tunes
  DistilBERT on the same SST-2. KEEP SST-2 as the single sentiment dataset through to Part C.
- Support-KB `corpus` (B5 semantic search) is an in-notebook 8-item list, not an external dataset.
- Carry-forward datasets from Part A still apply downstream: A1 used RAW
  `load_dataset('SetFit/bbc-news', split='train').to_pandas()` with `text_col='text'` /
  `cat_col='label_text'` (the pre-stemmed Dropbox `bbc.csv` is REJECTED, never reuse it); A3 used
  TWCS via `pd.read_csv(TWCS_URL dl=1, nrows=...)`.

### Pins / environment conventions (must stay consistent)
- `numpy<2` course-wide. B4 pins it even though Colab torch is numpy-2 compatible, purely for B5
  gensim/scipy consistency. Both notebooks carry the Colab "restart runtime after install" note
  because Colab preinstalls numpy 2.x.
- `scipy<1.13` whenever gensim is present (gensim 4.3 imports `scipy.linalg.triu`, removed in 1.13).
  B5 pins it; any later notebook that re-imports gensim must keep this pin.
- `gensim==4.3.3` (brings `gensim.downloader`; uses `key_to_index`/`index_to_key`, NOT the removed
  4.0 `vocab` attribute; OOV guard is `word in wv.key_to_index`).
- `transformers==4.57.1` (B5 hard-pins; do NOT let it resolve to 5.x, which forces numpy 2 and drops
  the summarization/QA/translation pipelines Part C needs). `sentence-transformers==3.4.1`,
  `datasets>=2.19,<3`.
- Final deliverable remains the guarded `gr.Interface` / `gr.Interface.from_pipeline` wrapped in
  `try/except ImportError`. Persona thread continues as "open the black box to beat the zero-shot
  ceiling."

### Scenario decisions to keep coherent
- Same customer-support-platform team and persona from A2/A3 carries through B4 and B5.
- The 70-85 percent zero-shot ceiling from A3 is the standing motivation; B5's homework
  LogisticRegression accuracy becomes the explicit bar-to-beat for the B7 MLP.
- PCA (not t-SNE) is the chosen 2D projection for small curated word sets (deterministic, no
  perplexity tuning); if any later notebook visualizes embeddings on a small set, stay with PCA for
  consistency.

---

## 4. Gaps / risks the next topic should be aware of

- RE-VERIFY B5 BEFORE BUILD. The B5 plan is flagged `[verify: FAIL]` in the handoff even though the
  three known lab-leak defects (Cell 20 spearmanr/cos_sim leak, Cell 22 util.semantic_search/top_k/
  [0] leak) are described as fixed in the plan text. Run `/verify-research` (or the verify skill) on
  B5 and confirm PASS before `/build-notebook`. B4 is `[verify: PASS]`.

- Document-vector dimension mismatch (the most likely coherence break). B5 ships a 100-d averaged-
  word2vec `doc_vector` (homework `X`) AND a 384-d SBERT path. The through-line narrative says the
  B7 MLP trains on "the 384-dim tensors" (B4 Cell 4, B5 Cell 31/through-line) but the B5 homework
  saves the 100-d `X, y`. B7 MUST pick one and state it: either train on the saved 100-d `X` (then
  the "384-dim" narrative in B4/B5 is slightly off and should be reconciled), or re-encode the SST-2
  sentences with `embedder` to 384-d and re-save. Do not let B7 silently inherit an ambiguous input
  width.

- transformers pin discipline. The Part A carry-forward already flagged that A2 hard-pinned
  `transformers==4.57.*` while A3 used a loose `>=4.40` that could resolve to 5.x and break. B5
  correctly hard-pins 4.57.1. B6/B7/Part C must keep the hard 4.57.x pin; never loosen to a range
  that can pull 5.x (numpy-2 + dropped pipelines).

- gensim/scipy fragility on Colab. The `numpy<2` + `scipy<1.13` + `gensim==4.3.3` combination
  requires a Colab runtime restart after the install cell because Colab preinstalls numpy 2.x. Both
  B5 install cell and markdown carry the restart note; any later gensim-using notebook must repeat
  it or students hit `ImportError: cannot import name 'triu'`.

- Device idiom seam. The A->B transition changed `device` from a pipeline int (`0 if cuda else -1`)
  to a `torch.device` object. B5 passes `device=str(device)` to `SentenceTransformer`. B6/B7 must
  consistently use the `torch.device` object for tensors/models and only convert to int when calling
  a HuggingFace `pipeline`. Mixed idioms will produce device-mismatch errors.

- B6 owes the deferred bricks. B4 deliberately DROPPED the DataLoader and nn.Module sections from the
  source notebook (migration map: old cells 6ea82449..c9b9e1be -> DROP) on the explicit promise that
  "they move to B6 where they are first used." B6 MUST actually introduce `nn.Module`, loss,
  optimizer, and `TensorDataset`/`DataLoader`, or that promise (made in B4 Cells 0, 3, 9, 17, 24) is
  broken. The B4 import cell also drops the `TensorDataset, DataLoader` import for the same reason.

- Through-line guarantees that cannot drift: Part C must deliver exactly `distilbert-base-uncased` +
  SST-2 (and `distilbert-base-cased-distilled-squad` + SQuAD for QA), and B6/B7 must actually reuse
  `all-MiniLM-L6-v2` / `wv` as the feature source. Substituting models anywhere downstream breaks
  the narrative spine the whole course is built on.

- Bias/honesty content is load-bearing, not decoration. B5's analogy section deliberately surfaces
  encoded social bias (`man : doctor :: woman : ?` returning a stereotype) and the input-word-
  exclusion fragility of analogies. If later notebooks revisit embeddings, keep that honest framing
  rather than reverting to the "king - man + woman = queen" hype.
