# Adversarial Review - Consolidated Cross-Pass Report

Two independent adversarial passes over all 11 course notebooks (exercise + solution):

- **Pass 1 - codex o3** (foreground, read-only): static adversarial read of each notebook.
  Outputs in `codex-reviews/01..11-*.md`.
- **Pass 2 - Opus workflow** (+ REAL .venv smoke-runs that actually execute the notebook,
  shrunk to tiny sizes): `REVIEW-A.md`, `REVIEW-B.md`, `REVIEW-C.md`.

A finding flagged by BOTH passes is marked HIGH-CONFIDENCE. The smoke-run results come from
real execution in the pinned `.venv` (numpy 1.26.4, transformers 4.57.1, torch 2.2.2).

Six criteria checked per notebook: (1) runs correctly per section, (2) pedagogical gaps,
(3) labs are real labs + no hint leaks, (4) solution complete + safety-nets collapsible/hidden,
(5) runs as a script, (6) numpy<2 pinned.

---

## Smoke-run results (real execution)

| NB | Exercise | Solution | First error |
|----|----------|----------|-------------|
| 1  | FAIL | pass | env: `en_core_web_sm` not provisioned (harness/Colab provides it) |
| 2  | FAIL | n/a  | summarization model `.bin`-only; torch 2.2.2 < 2.6 refuses load (tasks 1-4 ran) |
| 3  | FAIL | FAIL | `display()` not imported - REAL BUG (exercise + solution) |
| 4  | PASS | pass | clean (model notebook for the course) |
| 5  | -    | FAIL | solution verification hardcodes `(200,384)`, breaks under resize |
| 6  | FAIL | pass | Lab 4 loop `loss=None` unguarded, no safety-net - REAL crash |
| 7  | FAIL | pass | inline asserts fire before safety-net (G6) - all nets dead code |
| 8  | PASS | pass | clean run |
| 9  | PASS | n/a  | clean (only Part C exercise that smokes green) |
| 10 | FAIL | FAIL* | exercise: assert-before-net (G6); solution: Apple-Silicon MPS only |
| 11 | FAIL | FAIL* | exercise: top-level PATH gate crashes (G9); solution: MPS only |

\* NB10/NB11 solution failures are Apple-Silicon MPS artifacts (G8), not logic bugs - they pass on Colab GPU / pure CPU.

---

## GENERIC findings (course-wide - stated once, do NOT fix per-notebook)

### G1 - Safety-net cells leak the answer in plain sight  [HIGH-CONFIDENCE: codex + Opus, all 3 parts]
"SAFETY-NET (provided)" cells render as plain code right next to the lab, reproducing the
verbatim solution, with `metadata == {}` (no `<details>`, no collapse/skip). A student scrolling
normally sees the answer before attempting. Worst offenders: NB8 (all 4 capstone labs),
NB11 cell 24 (directly under the TIER-3 "no scaffolding" lab), NB3 cell 26 (reprints the whole
Lab 4 solution).
**Fix:** wrap every safety-net body in a collapsed `<details>` block (or skip-mark metadata) so it
is hidden until the student chooses to open it.

### G2 - Hint comments dictate the solution token-for-token  [HIGH-CONFIDENCE: codex + Opus, all 11]
`# YOUR CODE` comments name the exact call, kwargs, and indexing; labs reduce to transcription.
Often compounded by a demo cell printing the identical code one cell above the lab (NB6 c20,
NB8 c25), or the theory cell pre-printing the only number the lab "derives" (T5 lr 3e-4 in NB10/11).
**Fix:** reduce hints to intent, not mechanics; stop pre-printing the answer above the lab.

### G6 - Inline `assert ... is not None` crashes the exercise before the safety-net  [Opus smoke-runs; NEW]
A hard assert placed in the lab/verification cell fires against the untouched `None` placeholder
BEFORE the sibling safety-net runs, so the exercise CRASHES instead of degrading and the net is
unreachable dead code. Root cause of the NB6, NB7, NB10 exercise smoke FAILs.
**Fix:** move asserts AFTER the safety-net cell, or strip hard asserts from placeholder cells and
let the `if X is None` rescue fire first.

### G8 - Apple-Silicon MPS auto-dispatch breaks solution runs  [Opus smoke-runs; NEW]
HF Trainer/accelerate auto-sends the model to MPS even when the notebook sets `device="cpu"`,
raising `Placeholder storage has not been allocated on MPS device!` during train/generate.
Reproducibility hazard on Macs (passes on Colab GPU / pure CPU).
**Fix:** force CPU when not cuda (`TrainingArguments(use_cpu=True)` / disable MPS) or document the
Colab-GPU assumption.

### G3 - Provided verifications hardcode pre-shrink magic numbers  [Opus; NB5]
NB5 cell 25 asserts `(200,384)` / `len==200`; breaks when the subset size changes.
**Fix:** derive expected shapes from the data (`(len(s1),384)`), never hardcode.

### G7 - Broken diagram links  [Opus; NB5 only]
NB5 `../diagrams/...` links escape the notebook folder, 3 of 4 filenames are wrong, and
`solutions/2-Build-It/diagrams/` does not exist. (Part A and Part C use inline mermaid, so they
do not recur there.)
**Fix:** correct the paths/filenames; create the missing solutions diagrams dir or drop the links.

### G4 - numpy<2 correctly pinned everywhere  [HIGH-CONFIDENCE, GOOD]
All 11 pip cells pin `numpy<2` (many also pin transformers==4.57.1 / scipy<1.13 / gensim==4.3.3
and explain the Colab restart). No action.

### G5 - Labs are genuine and solutions are complete  [HIGH-CONFIDENCE, GOOD]
Every lab leaves real logic blank; every sibling solution fills it (often with "Common mistake"
notes). The build quality is good - the problems are answer-VISIBILITY and a few run bugs, not
bad teaching.

---

## SPECIFIC findings (per notebook - unique, real)

- **NB1:** TF-IDF used to name clusters but never explained; `min_df=3` empty-vocab risk on tiny
  subsets; cell 8 claims irregular lemmatization (`better->well`) the demo never shows; Labs 3.1/4.1
  have NO safety-net (inconsistent coverage).
- **NB2:** summarization model `sshleifer/distilbart-cnn-12-6` is `.bin`-only -> fails on the course
  .venv; intro promises "every cell runs on CPU" / "4.x pin works cleanly" - falsified by that fail.
  No in-notebook safety-nets at all.
- **NB3:** `display()` not imported (exercise + solution) - REAL bug. Wrap-up asserts "70-85% vs 95%"
  accuracy but NO metric is ever computed; `CONF_THRESHOLD=0.5` unjustified; promised sklearn
  metrics report never produced.
- **NB4:** cleanest notebook (no leaks, complete solution). Minor: names `torch.inference_mode()`,
  `.detach()`, `torch.mm` without demoing them.
- **NB5:** hardcoded `(200,384)` assert (G3); broken diagram links (G7); Spearman-vs-Pearson rationale
  only in the solution, not the exercise.
- **NB6:** Lab 4 training loop `loss=None` unguarded + no safety-net -> crashes (G6); Lab 5 silently
  no-ops when unfilled. Over-exposes (Labs 2/3) AND under-protects (Labs 4/5).
- **NB7:** 5 inline asserts make every safety-net unreachable (G6); cell 4 loads "B5 homework" file a
  standalone student does not have; no SentimentMLP forward demo before the build lab.
- **NB8:** worst G1 - all 4 capstone labs leak verbatim. Lab 3 framed "Tier 3 - no scaffolding" but
  the loop is printed one cell above.
- **NB9:** GOOD (only Part C exercise that smokes green). Lab 1 prompt says `padding=...max_length=32`
  but a prompt-faithful `padding=True` makes the pad-ratio ill-defined; the trap is only disclosed in
  the solution.
- **NB10:** assert-before-net crash (G6); MPS hazard (G8); Stretch B reuses the SQuAD-tuned model for
  summarization without a "load fresh t5-small" caveat.
- **NB11:** top-level `PATH=None` + assert gates the WHOLE notebook (G9) - crashes before any
  safety-net, breaking the cell-0 "finale always reachable" promise; cell 24 prints the answer under
  the Tier-3 lab; `max_input=256` silently truncates SQuAD gold spans; Lab B4 promises before/after
  but only calls the fine-tuned model.

---

## Verdict

The course is pedagogically strong, the single line holds, and `numpy<2` is pinned throughout -
but two systemic answer-VISIBILITY defects (G1 visible safety-nets, G2 token-for-token hints) and a
class of run-bugs (G6 assert-before-net, G9 top-level gate, NB3 display, NB6 unguarded loop) mean
several exercises crash on a straight run and most labs can be solved by copying. NB4 and NB9 are the
models to copy; NB8 and NB11 need the most work.

## Ranked fix plan

**Tier 1 - blocking run bugs (exercises that crash as shipped):**
1. NB11 G9: default `PATH` to a valid branch (or soft-warn) so safety-nets/finale are reachable.
2. NB6/NB7/NB10 G6: move inline asserts AFTER their safety-net cells (or drop them from placeholders).
3. NB3: `display()` -> `from IPython.display import display` (exercise + solution).
4. NB6: guard the Lab 4 loop body + add a Lab 4 safety-net.

**Tier 2 - answer visibility (the systemic teaching defect):**
5. G1: wrap ALL safety-net cells in collapsed `<details>` (highest: NB8 all labs, NB11 c24, NB3 c26).
6. G2: soften `# YOUR CODE` hints to intent; stop pre-printing answers in the demo/theory cell above.

**Tier 3 - correctness + reproducibility:**
7. NB5 G3: data-derive the hardcoded asserts; G7: fix/remove the diagram links.
8. G8: force CPU-when-not-cuda in NB10/NB11 (or document Colab-GPU requirement).
9. NB2: switch summarization to a safetensors model (e.g. `facebook/bart-large-cnn`) or note the pin.

**Tier 4 - honest claims:**
10. NB3 "70-85% vs 95%" (compute a metric or drop), NB9 Lab 1 padding mismatch, NB11 Lab B4
    before/after + 256-token truncation caveat, NB10 Stretch B fresh-model caveat, NB1 lemmatization
    claim, NB8 "no scaffolding" contradiction.

**No fixes applied yet - awaiting approval.**
