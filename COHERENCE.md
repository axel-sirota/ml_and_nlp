# Course Coherence Re-verification

One-line goal: the ML and NLP course must be ONE coherent line that builds, notebook 1
through notebook 11, into a working fine-tuned-transformer Gradio chatbot. This document
re-verifies that the spine holds end to end, traces the exact carried names at every seam,
and ranks every remaining coherence break.

Active course = the 11 notebooks under exercises/ (mirrored in Solutions/):
  1  1-Pre-NLP-Tools/1-NLP_From_Scratch.ipynb            (A1)
  2  1-Pre-NLP-Tools/2-Pipeline_Tour.ipynb               (A2)
  3  1-Pre-NLP-Tools/3-Capstone_A.ipynb                  (A3)
  4  2-Build-It/4-PyTorch_Tensors.ipynb                  (B4)
  5  2-Build-It/5-Word2Vec_and_Sentence_Embeddings.ipynb (B5)
  6  2-Build-It/6-PyTorch_NN_Basics.ipynb                (B6)
  7  2-Build-It/7-MLP_Word2Vec.ipynb                     (B7)
  8  2-Build-It/8-Capstone_B.ipynb                       (B8)
  9  3-Transfer-Chatbot/9-DistilBERT_Chatbot.ipynb       (C9)
 10  3-Transfer-Chatbot/10-EncoderDecoder_QA_Chatbot.ipynb (C10)
 11  3-Transfer-Chatbot/11-Capstone_C.ipynb              (C11)

## The single line 1 to 11

Each notebook is an independent Colab session, so handoffs are by CONCEPT, SKILL, and
re-declared NAME, not by live in-memory objects. There is therefore no cross-notebook
NameError risk anywhere: every prior artifact a notebook depends on is re-created locally
before use. The line is traced below by the EXACT carried names.

NB1 (A1) -> NB2 (A2)
  Carries: the embedder all-MiniLM-L6-v2 (named "the through-line of the course"); the
  spaCy nlp / doc.ents NER pattern; TextBlob(sample).words tokenization; the distilbert
  thread named as the eventual fine-tune target. Scenario switch BBC news -> customer
  support is acknowledged in A1's closing note AND re-confirmed in A2's intro.
  Handoff: explicit. A1's "Bridge to A2" names A2 by title and previews its exact tasks.

NB2 (A2) -> NB3 (A3)
  Carries: the zero-shot pipeline, built as router (pipeline("zero-shot-classification",
  default facebook/bart-large-mnli)); the ner pipeline with aggregation_strategy="simple";
  the gr.Interface.from_pipeline Gradio shell idea; distilbert-base-uncased as the Part C
  fine-tune target.
  Handoff: explicit and verified at the variable level. A2 names A3 by title; A3 openly
  flags the rename "This clf is the same zero-shot classifier you met in A2 (there it was
  the router)". Confirmed in source: A2 uses router (16x), A3 uses clf (10x), both on
  facebook/bart-large-mnli.

NB3 (A3) -> NB4 (B4)
  Carries: CONCEPTS only (correct, B4 is a from-scratch foundations reset). route_tweet +
  clf wrap a gr.Interface (verified) and A3 states "This is the SAME UX skeleton the Part C
  chatbot will use". The zero-shot 70-85 percent ceiling is the motivation handed forward.
  Handoff: explicit. A3's "Bridge to Part B" names B4; B4's intro reaches back accurately
  ("a zero-shot router that hit roughly 70-85 percent accuracy"), matching the stated ceiling.

NB4 (B4) -> NB5 (B5)
  Carries: SEED=42 and the torch.device idiom ("Same idiom as B4"); the (N,1,D)-(1,N,D)
  pairwise-distance broadcasting trick flagged as "the exact trick behind B5 similarity";
  the promise that sentences become 384-dim tensors from all-MiniLM-L6-v2.
  Handoff: explicit and confirmed. B5 loads embedder = SentenceTransformer("all-MiniLM-L6-v2")
  (verified, 4x in A1 / 3x in B5) plus wv = glove-wiki-gigaword-100 (the 100-d word-vector track).

NB5 (B5) -> NB6 (B6)
  Carries: SEED=42, device idiom, and the 100-d doc_vector width. B6 sets N_FEATURES=100
  annotated "same width as B5's doc_vector output (glove-wiki-gigaword-100)". The
  LogisticRegression baseline is named as the bar B7 must beat.
  Handoff: B6 substitutes a make_classification toy matrix that "stands in for the word2vec
  document vectors from B5" and tells the student it gets deleted in B7. CAVEAT: the literal
  load-bearing artifact (X.npy / y.npy / doc_vector / baseline) lives in B5's ASYNC Homework
  Extension, not the main body (verified: B5 has np.save 1x inside a Homework cell). B6's toy
  fallback means a student who skips the homework still flows in cleanly.

NB6 (B6) -> NB7 (B7)
  Carries: the nn.Module class SentimentMLP (100->128->2, Linear->ReLU->Dropout(0.3)->Linear,
  raw logits) and the six-line training loop. B6 frames these as "reused verbatim" in B7.
  Handoff: B7 REDEFINES SentimentMLP from scratch (Tier-2 lab) with the identical shape
  (verified: SentimentMLP 14x in B7) and uses its own train_one_epoch/evaluate instead of
  train_model. Conceptual reuse, correct for standalone notebooks; the "verbatim" wording
  slightly oversells literal code reuse.

NB7 (B7) -> NB8 (B8)
  Carries: wv = glove-wiki-gigaword-100, the doc_vector mean-pool helper, the
  LogisticRegression-baseline-then-beat-it pattern, the B6 training-loop machinery.
  Handoff: B7 saves b7_sentiment_mlp.pt + b7_scaler.pkl as "deployment artifacts" but B8
  loads NEITHER and retrains from the Hub. B8 also renames the class: B7 = SentimentMLP,
  B8 = MLPClassifier (verified: SentimentMLP 14x in B7, MLPClassifier 16x in B8, zero overlap).
  Dataset switch SST-2 -> IMDB is explicitly acknowledged in B8 ("brand-new dataset").

NB8 (B8) -> NB9 (C9)
  Carries: CONCEPTS only (the measured accuracy gap / the averaging ceiling). B8 saves NO
  file; C9 starts fresh from the Hub. B8's "Bridge to Part C" names C9 and the distilbert-
  base-uncased fine-tune. Dataset switch IMDB -> SST-2 is acknowledged in C9 ("same KIND of
  binary sentiment task ... so the comparison is fair").
  Handoff: explicit and honest ("You measured the gap; C9 closes it").

NB9 (C9) -> NB10 (C10)
  Carries: device, SEED=42, the HuggingFace Trainer recipe, save_pretrained -> SAVE_DIR ->
  from_pretrained reload, the guarded gr.Interface (try/except ImportError), the
  encoder-only / decoder-only / encoder-decoder diagram. C9 is verified BINARY sentiment:
  id2label POSITIVE/NEGATIVE on glue/sst2, distilbert-base-uncased, classify_message in
  gr.Interface (verified: POSITIVE 10x, NEGATIVE 10x, id2label 9x, glue + sst2 present).
  Handoff: explicit. C9's "Next: C10" names the encoder-decoder T5/BART jump into "this same
  Gradio shell". C10's prerequisites list C9 exactly.

NB10 (C10) -> NB11 (C11)
  Carries: build_input(question, context) returning "question: {q}  context: {c}", t5-small,
  max_input=256, max_target=32, LR=3e-4, SQuAD, the Seq2SeqTrainer + DataCollatorForSeq2Seq
  + predict_with_generate recipe, the guarded gr.Interface (two inputs -> written answer).
  Handoff: explicit. C10's "Next: C11" hands a clean two-path choice. C11 reproduces both
  paths. Verified: C11 contains build_input (12x) and the byte-identical "question: {question}
  context: {context}" format (2x), with an assert pinning the two-space format.

NB11 (C11)
  This IS the final Gradio chatbot. Path A re-derives the C9 shape (distilbert-base-uncased,
  glue/sst2, id2label {0:NEGATIVE,1:POSITIVE}, classify_message, Trainer +
  DataCollatorWithPadding). Path B re-derives the C10 shape (t5-small, build_input,
  answer_question, Seq2SeqTrainer + DataCollatorForSeq2Seq). Student fine-tunes a pretrained
  transformer on a justified public dataset, evaluates on a held-out split, saves+reloads via
  save_pretrained/from_pretrained (id2label persists through config.json), and ships inside
  the same guarded gr.Interface. PROVIDED safety-net cells guarantee the Gradio finale is
  reachable even if a lab is skipped.

VERDICT ON THE LINE: YES. One coherent line runs from notebook 1 into the shipped chatbot.
Every seam either carries a re-declared name that matches the prior notebook exactly, or
hands forward a clearly named CONCEPT with the scenario/dataset switch acknowledged in prose.
No seam has a NameError risk or a claimed-but-undefined live object. The chatbot end-goal is
reached at C9 (first shipped Gradio chatbot), generalized at C10 (generative), and assembled
unaided at C11 (the finale).

## Build-into-chatbot

Part A (NB1-3, "What NLP Can Do"): ships real NLP with ZERO training and plants the chatbot
vessel. A1 does NLP by hand (spaCy NER, TextBlob tokenize, sentence-embeddings + KMeans) so
the learner feels the plumbing. A2 collapses those tasks into one-line pipeline() calls and
introduces the zero-shot router, the distilbert fine-tune target, and gr.Interface. A3
(Capstone A) ships a confidence-thresholded zero-shot router (route_tweet + clf) wrapped in
gr.Interface on the real TWCS Twitter corpus, names the 70-85 percent zero-shot ceiling, and
states "This is the SAME UX skeleton the Part C chatbot will use". Hands forward: the Gradio
shell, the zero-shot skill, and the explicit motivation to fine-tune past the ceiling.
-> Hands to Part B: "to fine-tune a transformer you first have to understand what is inside
   one ... that is exactly Part B, starting with PyTorch tensors in B4."

Part B (NB4-8, "How It Works"): opens the black box and earns the training. B4 = tensors +
autograd (.backward(), the engine). B5 = text-as-geometry (100-d GloVe doc_vector for features
+ 384-d SBERT for semantic search / Part C). B6 = nn.Module / loss / optimizer / DataLoader /
six-line loop on a toy matrix. B7 = the "stopper" that snaps B5 features into the B6 model to
beat the LogReg baseline (embeddings-as-features). B8 (Capstone B) = re-derive the whole
pipeline solo on IMDB and MEASURE the hard ceiling of averaged static features ("not good"
looks positive). Hands forward: the measured ceiling and the .backward() engine.
-> Hands to Part C: "You measured the ceiling of averaged static features; C9 fine-tunes a
   contextual transformer with the SAME .backward() engine to break it, then ships it in Gradio."

Part C (NB9-11, "Transfer + Chatbot"): the payoff. C9 fine-tunes ENCODER-ONLY DistilBERT on
SST-2 to beat the Part B MLP and wraps it in a guarded gr.Interface (the chatbot is born). It
plants the architecture diagram. C10 fine-tunes ENCODER-DECODER t5-small on SQuAD so the bot
WRITES answers (same Trainer -> save_pretrained -> guarded-Gradio recipe, generalized to
Seq2SeqTrainer). C11 (Capstone C, the finale) forces the student to choose Path A (C9 shape)
or Path B (C10 shape), re-derive the recipe unaided, fine-tune on a public dataset, evaluate
honestly, and ship it in Gradio. C11 is the LAST notebook: it closes the arc ("That is the
course") and points beyond (RAG grounding, confidence-threshold abstention, EM/F1) rather
than to a next notebook.

END STATE: C11 ships a working fine-tuned-transformer Gradio chatbot. The chatbot is reached.

## Coherence breaks

Ranked by severity. Tag: notebook number; HARD = NameError / claimed-but-undefined /
unacknowledged dataset switch; SOFT = unacknowledged rename or thematic-seam slip.
NONE of the breaks below are HARD. There are zero NameError or claimed-but-undefined-object
breaks in the entire line.

1. [SOFT] NB10 (C10), scenario-flavor mismatch. C10 cell states the C9 DistilBERT chatbot
   "TAGS each incoming message: billing, bug report, praise", but C9 verifiably trains a
   BINARY sentiment classifier (id2label POSITIVE/NEGATIVE on glue/sst2). The 3-class
   billing/bug/praise taxonomy is never built in C9. Verified in source: C10 contains both
   "billing, bug report, praise" and "TAGS each incoming"; C9 contains POSITIVE/NEGATIVE
   only. Misdescribes the prior deliverable. No variable referenced -> NOT a runtime break.
   Highest-ranked because it is a factual misstatement about a prior notebook's output that a
   reader will notice.

2. [SOFT] NB8 (B8), silent rename presented as prior work. B8 intro/Scenario calls the B7
   model "the MLP ... we built together (B7)" and has the student build MLPClassifier(nn.Module),
   but B7's class is SentimentMLP, never MLPClassifier. Verified: SentimentMLP 14x in B7,
   MLPClassifier 16x in B8, zero overlap. No NameError (B8 rebuilds from scratch as a
   capstone), but the carried name does not match the claim.

3. [SOFT] NB7->NB8 (B7), dangling outbound artifacts. B7 saves b7_sentiment_mlp.pt and
   b7_scaler.pkl framed as deployment artifacts, but B8 loads neither and retrains from the
   Hub on a different dataset. The real carried artifacts (wv, doc_vector recipe, LogReg-
   baseline pattern) DO cross the seam; only the serialized files dangle. Pedagogically
   motivated, not a break.

4. [SOFT] NB5->NB6 (B5), load-bearing handoff gated behind optional homework. The literal
   word2vec handoff (X.npy / y.npy / doc_vector / LogReg baseline) is produced in B5's ASYNC
   Homework Extension, not the main body (verified: np.save appears only inside a Homework
   cell). B6 tolerates this with a make_classification toy fallback and B7 has a rebuild-from-
   wv+SST-2 fallback, so a student who skips it still runs cleanly; only the literal saved-file
   handoff is gated.

5. [SOFT] NB6 (B6), conceptual-vs-literal reuse framing. B6 says B7 reuses SentimentMLP and
   train_model "verbatim", but B7 redefines SentimentMLP from scratch and uses its own
   train_one_epoch/evaluate early-stopping loop instead of train_model. Correct for
   independent notebooks; wording oversells code-level reuse.

6. [SOFT] Course-wide naming looseness (B5/B6/B7/B8). B5 loads GloVe (glove-wiki-gigaword-100
   via gensim) but all four notebooks call the vectors "word2vec". Consistent across the seam,
   so it never causes downstream confusion. Sub-items: doc_vector tokenizer drift (B7 uses
   re.findall(r"[a-z']+", ...), B8 uses .split()); B8 stretch compares against the pre-
   finetuned distilbert-base-uncased-finetuned-sst-2-english while C9 fine-tunes base
   distilbert-base-uncased (different checkpoint, same family).

7. [SOFT] NB10/NB11, cross-module RAG homework references B5's semantic_search(query, top_k=3),
   defined in Part B, not in Part C. It is explicitly a "sketch/homework" prompt, not provided
   executable code, so no runtime break; a student running it literally would NameError. Low
   severity, flagged as homework.

8. [SOFT] Minor attribution slips. A1 loosely frames all-MiniLM-L6-v2 as "the thing you
   fine-tune" while A2/Part C actually fine-tune distilbert-base-uncased (both threads named,
   not contradictory). B4 credits all-MiniLM-L6-v2 to "A1" when it also recurs in the Part A
   capstone series. Cosmetic.

## Verdict + fixes

The course COHERES and BUILDS into the chatbot. ONE coherent line runs from notebook 1
through notebook 11 into a shipped, working fine-tuned-transformer Gradio chatbot. Every
seam carries either a re-declared name that matches the prior notebook exactly or a clearly
named concept with the scenario/dataset switch acknowledged in prose. There are ZERO hard
breaks: no NameError risk, no claimed-but-undefined object, no silent dataset switch. All
eight remaining issues are SOFT (prose/rename/framing) and none blocks the line or the
chatbot finale; the provided safety-net cells in C11 guarantee the Gradio chatbot is always
reachable.

Ranked fix list (all optional polish, none load-bearing):
1. NB10: fix the cell that says C9 "TAGS each incoming message: billing, bug report, praise"
   to describe C9's actual output (binary POSITIVE/NEGATIVE sentiment on SST-2).
2. NB8: rename MLPClassifier -> SentimentMLP (or change the "we built together (B7)" prose to
   say the student rebuilds it) so the carried name matches B7.
3. NB6/NB7: soften "reused verbatim" to "rebuilt with the same shape" to match the fact that
   B7 redefines the class and the loop.
4. NB7: either drop the b7_sentiment_mlp.pt / b7_scaler.pkl save framing as "deployment
   artifacts" or have B8 actually load them, so the saved-file seam is consumed.
5. NB5: move (or mirror) the X.npy/y.npy/doc_vector/baseline cell into the main lab body, or
   state plainly that B6/B7 fall back to a toy/rebuild path if the homework is skipped.
6. Course-wide: standardize "word2vec" vs "GloVe" wording (B5-B8) and align the doc_vector
   tokenizer between B7 and B8.
7. NB10/NB11: mark the semantic_search RAG snippet clearly as a Part B import the student must
   re-define, to avoid a literal NameError if copy-pasted.
8. NB1/NB4: tighten the all-MiniLM-L6-v2 attribution ("the embedder you saw in Part A",
   "the thing you fine-tune" -> distilbert).
