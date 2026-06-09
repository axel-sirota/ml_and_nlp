---
description: Research one ML&NLP course notebook and produce a cell-by-cell plan at plans/refactor/notebooks/<ID>-<slug>.md that /build-notebook can consume directly
---

Research and plan notebook: $ARGUMENTS

(Argument is a notebook ID from the refactor index, e.g. `A1`, `A2`, `B7`, `C10`.)

## GUARD: Read Course Manifest First

Before doing anything else, check for the course manifest and the refactor index:

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
ls plans/refactor/index.md 2>/dev/null
```

If `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` does NOT exist, stop immediately and say:

> `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` not found. Create the course manifest first. Every research-topic run requires it before proceeding.

If `plans/refactor/index.md` does NOT exist, stop and say the refactor plan is missing.

If both exist, read them fully and keep their contents in mind throughout this command.

---

This command is the **natural precursor to `/build-notebook`**. Its only job is to produce a
plan file at `plans/refactor/notebooks/<ID>-<slug>.md` so that `/build-notebook <slug> colab`
can build the exercise + solution notebooks cell by cell from it.

**This command does NOT create any `.ipynb` notebooks.** It produces markdown only.

**Web research is MANDATORY. Use the `/research` skill for all web searches in this command.**

## FAILURE DEFINITION

Exactly two outcomes: COMPLETE SUCCESS or COMPLETE FAILURE. No partial credit.

**COMPLETE SUCCESS** = all true:
- All 5 research cycles completed and visible in chat
- Each cycle invoked the `/research` skill (Skill tool, skill="research")
- PRE-WRITE SELF-CHECK printed in chat with all YES answers
- Plan file written to `plans/refactor/notebooks/<ID>-<slug>.md` with all required sections
- For TOUCHED notebooks: every cell tagged FROM-OLD (with source file + cell id) or FROM-NEW

**COMPLETE FAILURE** = any of:
- Fewer than 5 research cycles
- Any cycle skipped the `/research` skill
- PRE-WRITE SELF-CHECK not printed, or any field NO at Write time
- Plan file not written, or written with stub/placeholder content
- TOUCHED notebook plan missing the per-cell FROM-OLD/FROM-NEW provenance

If COMPLETE FAILURE: report what failed and restart from the beginning. Do not patch a partial result.

---

## Command Arguments

```
/run-research-topic <notebook_id>
```

Example: `/run-research-topic A2`

The notebook ID resolves against `plans/refactor/index.md`, which gives:
- the new file path and slug,
- the status tag (NEW / RENEWED / TOUCHED),
- the old source notebook (for RENEWED / TOUCHED),
- the per-notebook intent file in `plans/refactor/notebooks/<ID>-*.md`.

---

## Output Contract (NON-NEGOTIABLE)

1. **One file only**: `plans/refactor/notebooks/<ID>-<slug>.md` (overwrite the existing
   intent stub for that ID with the full cell-by-cell plan). Never write anywhere else.
2. **No `.ipynb` files.** No Write calls except to that one plan file.
3. **Plan must be directly executable by `/build-notebook`** - every cell has enough detail
   that the builder does not need to invent content.
4. **Structure** (every section required):
   - `# <ID> - <Title> - Cell-by-Cell Plan`
   - `## Status` (NEW / RENEWED / TOUCHED + source notebook path if any)
   - `## Context` (what students arrive with by exact variable/API name; the key insight they leave with)
   - `## Chatbot Through-Line` (how this notebook moves the course toward the final chatbot)
   - `## STAR Story` (Situation / Task / Action / Result framing for the notebook's scenario)
   - `## Deliverables` (exercise + solution paths)
   - `## Session Timing (~60-90 min)` table
   - `# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells)`
   - Each cell: `## Cell N - Markdown/Code: Description` with full content in a fenced block,
     AND a provenance line (see TOUCHED rule below)
   - `# VERIFICATION CHECKLIST`
   - `# RESEARCH VALIDATED (Month Year)`

### TOUCHED / RENEWED cell provenance (MANDATORY for non-NEW notebooks)

For NEW notebooks, every cell is authored fresh - tag each cell `Provenance: FROM-NEW`.

For TOUCHED and RENEWED notebooks you MUST open the **old source notebook** named in the
index, normalize/list its cell ids, and for EVERY planned cell add one line:

- `Provenance: FROM-OLD <old_file> cell <cell_id>` - reuse this old cell (verbatim or lightly
  edited; state which), OR
- `Provenance: FROM-NEW` - this cell is authored fresh for the new arc.

For FROM-OLD cells that change, add a `Change:` line saying exactly what to alter
(e.g. "swap TF-IDF features for gensim word2vec", "retheme story to chatbot").
For TOUCHED notebooks, also add a `## Cell Migration Map` table near the top listing
old cell id -> new cell N -> action (keep / edit / drop) so the builder can diff against
the original notebook deterministically.

### Writing the Plan File in Batches

If the plan will exceed ~150 lines, write the header + timing first, then append
cell-by-cell content in ~50-line chunks with Edit, then append the checklist + research block.

---

## Pre-Work: MANDATORY Reading

Before any research, read:

1. `CLAUDE.md` - teaching philosophy, notebook conventions, PyTorch/HF stack, tone
2. `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` - course manifest (stack, datasets, models, arc)
3. `plans/refactor/plan.md` and `plans/refactor/index.md` - the new arc + this notebook's role
4. `plans/refactor/notebooks/<ID>-*.md` - this notebook's intent stub (topics already decided)
5. `initial_docs/outline.md` - extract this notebook's section (by ID heading)
6. `.claude/commands/build-notebook.md` - the consumer of your plan (structure it expects)
7. For RENEWED / TOUCHED: open the **old source notebook** from the index - list exact cell
   ids, variable names, datasets, and helpers so you can write the provenance map
8. For ID > A1: open the **prior notebook in the arc** - list the exact variables, model
   objects, and helpers students already have so this notebook bridges cleanly

---

## MANDATORY CYCLE TRACKER

Post this at the start of research and update after each cycle. Do NOT write the plan file
until all 5 boxes are `[x]` AND all show `/research invoked: YES`.

```
CYCLE TRACKER
[_] Cycle 1 - Arc Alignment and Prior-Notebook Continuity        /research invoked: NO
[_] Cycle 2 - Environment and Dependencies (Colab + HF/PyTorch)  /research invoked: NO
[_] Cycle 3 - Pedagogical Structure (Theory -> Demo -> Lab)      /research invoked: NO
[_] Cycle 4 - Lab Difficulty and Stretch / Homework             /research invoked: NO
[_] Cycle 5 - Chatbot Through-Line, Timing, Take-Homes          /research invoked: NO
```

A cycle is NOT complete if `/research invoked` still says NO. No exceptions.

---

## Process: 5 TDD-Style Research Cycles

**All cycles visible in chat.** Every cycle MUST invoke the `/research` skill (skill="research").
Raw WebSearch / WebFetch / "I already know this" do NOT count.

### Cycle 1 - Arc Alignment and Prior-Notebook Continuity
- Confirm this notebook's topics against its intent stub + the outline.
- For RENEWED/TOUCHED: open the old source notebook; build the cell id inventory.
- For ID > A1: open the prior notebook; list carried-over variables/model objects/helpers.
- Default model IDs for the course: encoder-only `distilbert-base-uncased`; seq2seq
  `t5-small` or `facebook/bart-base` (unless the manifest says otherwise). NO LLM API keys.
- **Invoke `/research`**: current best practices for this notebook's core concept with
  PyTorch + HuggingFace `transformers`/`datasets` as of 2026.
- **Refutation**: any variable-name collision with prior notebooks? Re-implementing
  something already built earlier in the arc?

### Cycle 2 - Environment and Dependencies (Colab + HF/PyTorch)
- Lock the full `!pip install -q` line, pinned to current stable; always include `numpy<2`.
- Standard setup: GPU check `torch.cuda.is_available()`, `device = ...`, `torch.manual_seed(42)`.
- Verify dataset access (HuggingFace `datasets` ids, or the reference Dropbox CSV URLs).
  Course datasets: HF `datasets`, SST-2 (GLUE/SST-2), Yelp reviews, SQuAD for Q&A.
- Confirm import paths against current docs (transformers/datasets/sentence-transformers/
  gensim/spaCy APIs drift - verify exact syntax).
- **Invoke `/research`**: current stable versions + correct import paths; any `numpy<2` conflict.
- **Refutation**: deprecated import paths? version conflicts?

### Cycle 3 - Pedagogical Structure (Theory -> Demo -> Lab)
- For EACH concept (2-4 per notebook), plan cells in order following the repo's rule:
  never chain more than 3 markdown/theory cells without a code cell showing it in action.
  - **Theory** markdown: section header + first-person "here's the problem" setup.
  - **Demo** code: complete, runnable, heavily commented (what AND why).
  - **Lab** markdown: step-by-step instructions.
  - **Lab starter** code: `None # YOUR CODE` placeholders (exercise) with verification.
- Lab tiers: most are guided (15-20 min); at most ONE harder stretch lab per notebook.
- **Invoke `/research`**: common gotchas when doing this demo/lab with the chosen library.
- **Refutation**: does any `# YOUR CODE` comment leak the solution? Is the demo actually runnable?

### Cycle 4 - Lab Difficulty and Stretch / Homework
- Each in-class lab: standard (15 min) + clearly labeled stretch for fast finishers + a
  Homework Extension (async, deeper, production-oriented).
- Cover-the-solution test: from the starter comments alone, can a student solve it in
  under 30 seconds? If yes, hints are too strong - rewrite.
- Identify safety-net cells: any lab whose output variable a later cell consumes.
- **Invoke `/research`**: production concerns / edge cases that make a good homework extension.
- **Refutation**: is the stretch genuinely harder, or just more volume?

### Cycle 5 - Chatbot Through-Line, Timing, Take-Homes
- State how this notebook advances the course's end goal (a fine-tuned-model **Gradio chatbot**):
  what skill/artifact it contributes, and the one-line bridge to the next notebook.
- Verify total timing fits ~60-90 min.
- Produce RESEARCH VALIDATED block: every source URL + the specific fact extracted.
- Produce the VERIFICATION CHECKLIST.
- **Invoke `/research`**: most important production best practices for this topic for take-homes.
- **Refutation**: any unsourced claim? any unverified version or dataset id?

---

## HARD GATE: Pre-Write Self-Check (MANDATORY before any Write call)

Print this in chat and verify every line passes. Any NO = do not write; fix it first.

```
PRE-WRITE SELF-CHECK
--------------------
Cycle tracker all 5 checked [x]:              YES / NO
All 5 /research invocations confirmed:        YES / NO
Plan path is plans/refactor/notebooks/<ID>-*: YES / NO
No .ipynb files created:                      YES / NO
STAR story section present:                   YES / NO
Chatbot through-line section present:         YES / NO
TOUCHED/RENEWED: every cell has provenance:   YES / NO / N-A
TOUCHED: cell migration map present:          YES / NO / N-A
RESEARCH VALIDATED block has URLs:            YES / NO
VERIFICATION CHECKLIST present:               YES / NO
All cells have full content (not stubs):      YES / NO
AI-tells scan passed (no em/en dashes, no emoji): YES / NO
```

---

## Non-Negotiables (Violating Any = Complete Failure, Start Over)

1. Path is always `plans/refactor/notebooks/<ID>-<slug>.md` - never anywhere else.
2. No `.ipynb` files created.
3. `numpy<2` in every install cell.
4. `None # YOUR CODE` comments never reveal the answer.
5. Every concept has a Demo cell AND a Lab cell; max 3 theory cells without a code cell.
6. Every in-class lab has a stretch version and a homework extension.
7. All dataset ids / URLs verified before finalizing.
8. Prior-notebook continuity confirmed by opening the actual prior notebook.
9. For TOUCHED/RENEWED: open the old notebook; every planned cell tagged FROM-OLD (with
   file + cell id) or FROM-NEW; TOUCHED also gets a cell migration map.
10. STAR story + chatbot through-line sections both present.
11. All 5 cycles visible, all `[x]`, all `/research invoked: YES` before any Write.
12. EVERY CYCLE MUST INVOKE THE /research SKILL (skill="research"). No substitutions.
13. PRE-WRITE SELF-CHECK printed with all YES (or N-A where noted) before Write.
14. ZERO AI-TELLS in the plan file: no em dashes, no en dashes, no Unicode multiplication,
    no emoji. Plain ASCII only. Final pass before Write.

---

## Course Stack Defaults (this repo)

- **Framework**: PyTorch + HuggingFace (`transformers`, `datasets`, `evaluate`, `accelerate`,
  `sentence-transformers`), gensim, spaCy. NO TensorFlow/Keras. NO RNN/LSTM in core.
- **Environment**: Colab (GPU optional). `torch.manual_seed(42)`, device auto-select.
- **Models**: encoder-only `distilbert-base-uncased`; seq2seq `t5-small` / `facebook/bart-base`.
  No external LLM API keys; everything runs from the HF Hub.
- **Datasets**: public only - HuggingFace `datasets` (e.g. `imdb`, `glue/sst2`, `yelp_review_full`,
  `squad`), plus the course Dropbox CSVs. Q&A uses SQuAD.
- **Chatbot**: simple Gradio app loading the fine-tuned model (the course's end deliverable).
- **Placeholders**: `None # YOUR CODE` (exercise); full code + explanation (solution).

---

## Handoff

Once the plan file is written, end with:

> Plan written to `plans/refactor/notebooks/<ID>-<slug>.md`.
>
> Next step: run `/build-notebook <slug> colab` to generate the exercise + solution
> notebooks from this plan, 5 cells at a time with approval between batches.

Do not offer to run `/build-notebook` yourself.

---

## Notebook Edit Protocol (awareness)

If this command edits notebook cells (it should not - it writes markdown only), follow the
canonical procedure: normalize cell ids, locate cells by id + content, read back and assert
after each edit. Blind bulk index-based rewrites are forbidden.
