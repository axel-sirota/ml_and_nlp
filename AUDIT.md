# Pedagogy Audit

Date: <!-- YYYY-MM-DD -->

Course one-line goal: teach the tools -> build with them -> transfer the skill -> ship a Gradio chatbot. Every notebook must advance a single pedagogical line from classical NLP by hand (A1) to a fine-tuned transformer shipped in a Gradio UI (C9/C10/C11).

Scope: 11 notebooks across three Parts. Part A (A1, A2, A3): NLP by hand -> pipeline tour -> zero-shot router. Part B (B4, B5, B6, B7, B8): tensors -> embeddings -> nn.Module -> MLP -> capstone. Part C (C9, C10, C11): fine-tune DistilBERT -> encoder-decoder T5 -> assemble-unaided capstone with Gradio ship.

---

## Single pedagogical line A -> B -> C

Verdict: YES, one coherent line runs from A1 to the C11 Gradio chatbot, carried by exact named artifacts. The line is intact but has a handful of NAMING SEAMS and one genuine BROKEN PROMISE that should be reconciled.

Trace by exact carried names:

- A1 plants the seed artifact `embedder = SentenceTransformer("all-MiniLM-L6-v2")` and names it as the through-line reused in B and C. STAR scenario: BBC media-analytics / day-one unlabeled text. Forward bridge to A2 is explicit by name.
- A1 -> A2 SEAM (thematic only): A2 reframes the STAR from BBC media-analytics to a customer-support-ticket platform and shares NO concrete variables (nlp, bbc, embedder) with A1. Continuity is by task (sentiment/NER/zero-shot/QA), not by variable. A2 does reciprocate A1 by name in cell 0 ("In A1 you did NLP by hand...").
- A2 establishes `pipeline()`, `candidate_labels`, `aggregation_strategy="simple"`, the `{sequence, labels, scores}` result shape, `device`, and a Gradio preview (`gr.Interface.from_pipeline`). Forward bridge to A3 is explicit by name.
- A2 -> A3 SEAM (silent rename): A3 renames A2's zero-shot pipe `router` -> `clf` and `candidate_labels` -> `CANDIDATE_LABELS` (longer label strings). Intentional per A3 plan, but the notebook never says "this is the same router from A2". A3 introduces `clf`, `CANDIDATE_LABELS`, `reply_map`, `route_tweet`, `CONF_THRESHOLD`.
- A3 -> B4: A3 closes with the 70-85 percent zero-shot ceiling and bridges to Part B / PyTorch tensors. B4 opens on that exact number. Carried: `device`, `SEED=42`, and the named `all-MiniLM-L6-v2` embedder.
- B4 -> B5: carries `SEED=42`, `torch.manual_seed(SEED)`, `device`, and `embedder = SentenceTransformer("all-MiniLM-L6-v2")` verbatim. No variable collision (B4 uses scratch x/w/y; B5 adds wv, embedder, corpus, corpus_emb).
- B5 -> B6 -> B7 -> B8: carries `wv` (glove-wiki-gigaword-100, 100-d), `doc_vector` with OOV guard `word in wv.key_to_index`, `X`/`y`, the LogisticRegression baseline, `SEED=42`, `device`. B5 saves X/y via np.save (filenames inferred, not stated) and B7/B8 expect exactly X.npy/y.npy.
- B6 -> B7 BROKEN PROMISE (genuine break, not just a seam): B6 repeatedly promises "you keep this exact MLPClassifier and this exact loop" and "In B7 you call train_model on the word2vec X, y." B7 does NOT do this: it defines a NEW class `SentimentMLP` (Linear(100,128)->ReLU->Dropout(0.3)->Linear(128,2)) versus B6's `MLPClassifier` (in_dim->64->n_classes, dropout 0.2), never imports or calls `train_model`, and writes its own loop with StandardScaler + early stopping. The carry-forward names (X/y/doc_vector/baseline) hold, but the class and helper named as the artifacts are silently re-specced.
- B7 -> B8 SEAM: same architecture reintroduced under a different class name (B7 `SentimentMLP` vs B8 `MLPClassifier`).
- B8 -> C9: B8 measures the averaged-feature ceiling on IMDB and bridges to C9 fine-tuning distilbert-base-uncased + Gradio. C9 reciprocates verbatim. CORPUS SEAM: B8 trains on IMDB; C9 fine-tunes on SST-2. Both binary sentiment, conceptual line unbroken, but the exact corpus is not carried.
- C9 -> C10: carries `SEED=42`, `device`, `AutoTokenizer`, `save_pretrained`/`from_pretrained` (SAVE_DIR ./distilbert_sst2_final), guarded `gr.Interface`. Architecture arc encoder-only -> encoder-decoder stated explicitly.
- C10 -> C11: carries `build_input(question, context)`, `answer_question(question, context)`, `t5-small`, `Seq2SeqTrainingArguments`/`Seq2SeqTrainer`, `squad`, plus C9 Path A names. C11 is the finale; bridge is to PRODUCTION (abstention, drift, RAG), correctly NO next notebook.
- C11 SEAMS (minor): `classify_message` reused by NAME from C9 but with a different signature (C11 returns a (label, confidence) tuple via manual softmax; C9 returned a string via `chat_pipe = pipeline(...)`). C10's `chat_fn`/`save_dir` standardized to `gradio_fn`/`SAVE_DIR` in C11.

Bottom line: the single line reaches the Gradio chatbot. Fix the B6->B7 broken promise (the only real break) and acknowledge the A1->A2 scenario switch, the A2->A3 rename, and the B8->C9 corpus switch, so the story reads as literally continuous rather than rhetorically continuous.

---

## The beat (theory -> diagram -> demo -> lab -> safety-net)

Most notebooks comply with theory -> diagram -> demo -> lab. The SYSTEMIC, course-wide defect is the SAFETY-NET beat: almost no notebook supplies a working fallback value when a lab is skipped. The common pattern (`if <var> is not None:`) is a verification SKIP-GUARD, not a safety-net: it only suppresses the print, it never assigns a usable value, so any skipped lab whose output feeds a later cell bricks the run.

| Notebook | Beat OK? | What is broken |
|----------|----------|----------------|
| A1 | Mostly | No real safety-net anywhere; guards mistaken for safety-nets. Lab 2.1 (preprocess, cell 12) feeds cells 13 and 22 with NO fallback -> skip bricks the run. |
| A2 | Yes | Lab 2 has no standalone runnable swap-demo between theory and lab (demo is a non-executed snippet). Self-guarding labs acceptable here because no lab output is consumed downstream. |
| A3 | No (safety-net) | Section 5 has NO standalone demo cell before Lab 4 (NER demo lives only inside markdown). All 5 chained labs lack safety-nets; skip of Lab 1/2 crashes the rest. |
| B4 | Yes | No lab has a real safety-net, but harmless: every lab is self-contained, no output consumed downstream. |
| B5 | Mostly | Doubled diagrams (external .mmd link AND inline mermaid per concept). PCA demo has no preceding theory markdown. No safety-nets; verifications hard-assert non-None -> skip halts. |
| B6 | No (safety-net) | Diagrams + demos clean, but Labs 2/3/4/5 chain by shared mutable vars with NO fallback; skip of Lab 2 or 3 crashes the back half. |
| B7 | No (safety-net) | THE STOPPER. theory->diagram->demo strong but uses hard asserts, zero safety-nets; skipped Lab 2/3 bricks train/eval/save. Training-loop diagram (cell 20) placed AFTER training ran. `doc_vector` only defined in else-branch -> NameError on load path. |
| B8 | No (safety-net) | Clean cadence, but Labs 1/2/4 feed later cells with no fallback; skip crashes baseline.fit / .parameters() / metric subtraction. |
| C9 | No (chaining) | Section 1 chains FOUR markdown cells before first code (violates max-3 rule). Only 1 of 4 concept sections has a real lab; "Lab 2" is a commented-out stretch, not a lab. |
| C10 | Mostly | Concept 3 has no standalone demo before Lab 2 (mitigated by near-complete starter). All 3 labs feed later cells with no safety-net. |
| C11 | By design | Capstone intentionally has no per-concept demo (demos were C9/C10). Universal safety-net absence is the plan's deliberate "assemble unaided" design but leaves the finale brittle. Save/reload diagram misplaced inside Path B section. |

Beat verdict: most notebooks follow theory->diagram->demo->lab. Two true beat breaks beyond safety-nets: A3 Section 5 and C10 Concept 3 are MISSING-DEMO; C9 Section 1 is a 4-markdown CHAIN violation. The safety-net beat is absent in 8 of 11 notebooks and is load-bearing wherever a lab output feeds downstream.

---

## Lab tiers PER PART

Target per section: about 1 Tier-3 (hard, no guidance), about 2 Tier-2 (less guidance, not copy-paste), the REST Tier-1 (fill a one-liner from the demo).

### Part A (A1, A2, A3)

- A1: labs 3, tiers {1:1, 2:2}. A2: labs 2, tiers {1:1, 2:1}. A3: labs 5, tiers {1:2, 2:3}.
- Part A totals: Tier-1 = 4, Tier-2 = 6, Tier-3 = 0.
- Judgment: UNDER on Tier-3. Across 10 labs there is NOT ONE Tier-3. Part A is a tour-and-warm-up part, so a missing Tier-3 in A1/A2 is defensible, but A3 (the zero-shot router, the part's deliverable) should carry one Tier-3.
- Re-tier: promote A3's hardest step (e.g. building `route_tweet` end to end, or the NER `extract_entities` filter in Lab 4) to Tier-3 with a bare signature and no in-cell step list.

### Part B (B4, B5, B6, B7, B8)

- B4: labs 5, tiers {1:3, 2:2}. B5: labs 3, tiers {1:1, 2:2}. B6: labs 5, tiers {1:3, 2:2}. B7: labs 5, tiers {1:3, 2:2}. B8: labs 6, tiers {1:3, 2:3}.
- Part B totals: Tier-1 = 13, Tier-2 = 11, Tier-3 = 0.
- Judgment: UNDER on Tier-3 (zero across 24 labs in 5 notebooks) and slightly HEAVY on Tier-2. Most damning: B7 is THE STOPPER (the convergence climax of Parts A+B) yet its core steps (define/train the MLP) are Tier-1 one-liners; only Lab 3b and Lab 4 reach Tier-2. B8 is the capstone and also has no Tier-3.
- Re-tier: (1) In B7, promote defining/training `SentimentMLP` to Tier-2/3 by making the class itself a lab (currently a provided demo) and dropping the step list. (2) In B8, promote the train-loop lab (Lab 3) or the eval lab (Lab 4) to Tier-3. (3) B5 Lab 1 is Tier-1 near copy-paste (membership test + most_similar shown verbatim in demo); upgrade by requiring OOV handling inside the analogy. One Tier-3 per notebook would lift Part B from 13/11/0 toward a healthier per-section shape.

### Part C (C9, C10, C11)

- C9: labs 2 (effectively 1 real), tiers {1:1, 3:1}. NOTE: the "Tier-3" is the commented-out freeze-backbone stretch that never executes and has no None # YOUR CODE, so it is NOT a real lab; effective real labs = 1 (Tier-1). C10: labs 3, tiers {1:1, 2:2}. C11: labs 12, tiers {1:4, 2:7, 3:1}.
- Part C totals (as reported): Tier-1 = 6, Tier-2 = 9, Tier-3 = 2. Effective (dropping C9's phantom Tier-3): Tier-1 = 6, Tier-2 = 9, Tier-3 = 1.
- Judgment: C9 is UNDER and mis-counted: a 60-90 min transfer-learning notebook with effectively ONE Tier-1 lab; the fine-tuning core (the pedagogical center) is pure copy-run with zero student code. C11 is OVER on labs (12) and HEAVY on Tier-2 (7) but UNDER on Tier-3 for a finale: only the write-up (cell 23) is Tier-3; every CODE lab is Tier-1/2 with in-cell step lists, undercutting the "assemble unaided" intent. C10 is roughly on target.
- Re-tier: (1) C9: add a real Tier-2 lab on the fine-tuning pipeline (student writes `tokenize_fn` or the `rename_column` step) and stop labelling the commented freeze block "Lab 2". (2) C11: drop the in-cell step list from one core code lab (`classify_message` or `answer_question`) to a bare signature to create a genuine Tier-3 code lab.

Tier verdict overall: every Part is UNDER on Tier-3 (A: 0, B: 0, C: 1 effective) and the course is Tier-2-heavy. The biggest tiering defects are at the two climaxes: B7 (the convergence stopper) and C9 (the transfer-learning center) are the lightest where they should be the hardest.

---

## Safety-net gaps

Labs whose output FEEDS a later cell but have NO safety-net (these are build-breakers if the student skips the lab):

A1
- Lab 2.1 preprocess (cell 12) -> feeds cell 13 (avg_len) and cell 22 (TfidfVectorizer.fit_transform). Skip -> bbc["clean"] all-None -> both crash.

A3 (all 5 labs chain by shared variable; 4 are load-bearing)
- Lab 1 (8f38509d) -> Lab 2 does labeled["clean"] = ... on possibly-None labeled.
- Lab 2 (4ac30edc) -> feeds Lab 3.
- Lab 3 (37a4cd91) -> sets routing_set["route"]/["confidence"] = None; later cells index/format them (e.g. row["confidence"]:.2f raises on None).
- Lab 4 (249bf372) -> ner/extract_entities consumed by packaging route_tweet.

B6
- Lab 2 MLPClassifier (5cb1e329) -> instantiated by Lab 3, Lab 4, train_model demo, Lab 5. Skip cascades through the whole back half.
- Lab 3 outputs train_loader/model/loss_fn/optimizer (31007386) -> consumed by Lab 4 (808d70c2).

B7 (THE STOPPER; all four output-producing labs feed downstream)
- Lab 1 (cell 9), Lab 2 DataLoaders (cell 13), Lab 3a model (cell 16), Lab 3b train (cell 17) -> feed eval/save chain. Hard asserts halt on skip; downstream NameError/None-crash. Also predict_sentiment (cell 23) and Lab 4 (cell 26) NameError if doc_vector lands on the load path (defined only in else-branch).

B8
- Lab 1 (cell 13) -> cell 15 baseline.fit(None) crashes.
- Lab 2 (cell 20) -> cell 23 None.parameters() crashes.
- Lab 4 (cell 27) -> cell 29 trans_acc - None crashes.
- (Lab 3 cell 25 degrades to random-init garbage rather than crashing; a re-train fallback keeps the capstone honest.)

C10 (all three core labs feed later cells)
- Lab 1 (2fda217bf0fb) -> .map in cell 3d080c33e0f1.
- Lab 2 (2a3f773d7988) -> training; skip leaves model un-tuned for save sanity-check.
- Lab 3 (0403255f78fa) answer_question -> save sanity-check and Gradio finale. Gradio guards only ImportError, so a skipped Lab 3 launches the UI and errors at call time.

C11 (capstone; every lab output feeds the shared ship)
- All Path A labs (A1-A5) and Path B labs (B1-B4) -> shared save/reload (92105a6e) and Gradio ship (e1cad9ef). Assert-not-None gates; one missed lab halts the finale. Intentional per plan, but the intent should be stated in cell 0.

NOT build-breakers (self-contained, listed for completeness): A2 (no lab output consumed downstream), B4 (self-contained labs), B5 (halts on its own assert but nothing downstream consumes), C9 Lab 1 (lab_sentences/lab_enc/real_counts not consumed; pipeline rebuilds independently).

---

## Ranked fixes

Priority 1 - build-breakers (a skipped lab bricks the notebook):
1. B7 (THE STOPPER): add guarded safety-nets supplying working values for Lab 1, Lab 2 (DataLoaders), Lab 3a (model), Lab 3b (train loop) so eval/save/ship still run. Also define doc_vector UNCONDITIONALLY outside the if/else in cell 4 (fixes load-path NameError in cells 23 and 26).
2. B8: add `if <var> is None:` fallback cells after cells 13, 20, 27 (and a re-train fallback after 25) so baseline.fit, .parameters(), and the metric subtraction all run end to end.
3. B6: add guarded fallback definitions for MLPClassifier (after Lab 2) and for train_loader/model/loss_fn/optimizer (after Lab 3); replace Lab 4's bare `assert acc > 0.85` with a guarded warning.
4. A3: add fallback cells/in-cell guards seeding labeled, routing_set, the route/confidence columns, and ner/extract_entities, so skipping any of Labs 1-4 does not crash the rest.
5. A1: add a safety-net after cell 13 that detects an empty/None clean column and supplies a reference preprocess so cells 13 and 22 survive a skipped Lab 2.1.
6. C10: add safety-nets after Lab 1, Lab 2, Lab 3 (working preprocess, training args + trainer.train(), working answer_question) so the Gradio finale is reachable; the ImportError-only Gradio guard does not cover the skipped-lab path.
7. C11: either add safety-nets to the chained labs, OR state explicitly in cell 0 that the asserts are intentional gates (not hints) for this "assemble unaided" capstone.

Priority 2 - true beat breaks (missing demo / chained markdown):
8. C9: split Section 1's four-markdown chain - move one inline diagram to after the tokenizer demo (or merge the two theory mds) to satisfy the max-3-markdown rule.
9. A3: add a 2-line runnable NER demo cell before Lab 4 (reveal only the call shape, not the keep-groups filter) so Section 5 has the missing demo beat.
10. C10: add a tiny demo of Seq2SeqTrainingArguments before Lab 2, or explicitly note the C9 Trainer call is the reference demo.
11. A2: add a 2-line runnable model-swap demo before Lab 2 to tighten the theory->demo->lab beat.

Priority 3 - through-line reconciliation:
12. B6 -> B7 broken promise: either rename B6's class/helper to match B7 (SentimentMLP, hidden=128, dropout=0.3) AND align hyperparameters, or soften B6's "this exact class/loop" recap to "this same pattern" so the promise is truthful.
13. A2 -> A3 rename: add one sentence in A3 Section 4 theory noting clf is the same zero-shot pipeline as A2's router, just renamed with sharper labels.
14. A1 -> A2 scenario switch: add one sentence in A1's bridge or A2's intro acknowledging the BBC -> support-ticket scenario switch.
15. B8 -> C9 corpus switch: add a one-line note in B8 cell 32 (or have C9 subsample IMDB) so the IMDB -> SST-2 dataset thread is acknowledged.
16. C11: reconcile classify_message contract drift vs C9 (tuple vs string) in the comment/plan so the "verbatim reuse" claim is accurate.
17. B5: make np.save filenames explicit (X.npy, y.npy) so the B5 -> B7/B8 handoff is unambiguous.
18. B7 / B8 naming: align the model class name (SentimentMLP vs MLPClassifier) or note the re-derivation.

Priority 4 - tiering (raise difficulty at the climaxes):
19. B7: promote defining/training the MLP to a Tier-2/3 lab (make SentimentMLP a lab, drop the step list).
20. C9: add a real Tier-2 lab in the fine-tuning pipeline (tokenize_fn or rename_column) and stop labelling the commented freeze block "Lab 2".
21. C11: drop the in-cell step list from one core code lab (classify_message or answer_question) to a bare signature for a genuine Tier-3.
22. A3: promote route_tweet or extract_entities to Tier-3. B5: upgrade Lab 1 with OOV handling. B8: promote Lab 3 or Lab 4 to Tier-3.

Priority 5 - hygiene:
23. B5: de-duplicate doubled diagrams (pick external .mmd OR inline mermaid per concept). Coerce float(s) in Lab 1's isinstance assert (gensim returns numpy.float32). Add a PCA theory markdown cell.
24. A3: remove the unused `from sklearn.metrics import classification_report` (dead code that contradicts the no-accuracy framing).
25. B7: move the training-loop diagram (cell 20) before the loop it illustrates. C11: move the save/reload diagram next to the shared save cell (92105a6e).
26. Reconcile stale plan VERIFICATION CHECKLIST cell-count items (A1 23->27, B4 25->31, B7 ->31, B8 25->34) that drifted because inline mermaid diagram cells were added; the added diagrams are an improvement, the checklists are stale.
