---
description: AUTONOMOUS build of one notebook's exercise + solution from its plan, no human checkpoints. For use inside workflows/subagents. Colab + HuggingFace, writes under refactored/<PART>/.
---

Build notebook autonomously: $ARGUMENTS

(Argument: a notebook ID from plans/refactor/index.md, e.g. `A1`, plus its plan path and the
target part folder. The orchestrator passes these explicitly.)

This is the AUTONOMOUS variant of `/build-topic-notebook`. There is NO human in the loop:
do NOT ask for approval, do NOT stop every 5 cells for confirmation, do NOT invoke /save-state.
Build the full exercise notebook, then the full solution notebook, then validate. Report at the end.

## GUARD

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```
If missing, stop and report the manifest is missing. Otherwise read it and keep it in mind.

## Inputs (provided by the orchestrator)

- `ID` - notebook id (e.g. A1)
- `PLAN` - path to the cell-by-cell plan, e.g. `plans/refactor/notebooks/A1-nlp-from-scratch.md`
- `SLUG` - e.g. `nlp-from-scratch`
- `PART` - `A`, `B`, or `C`
- Output paths:
  - Exercise: `refactored/<PART>/exercises/<SLUG>.ipynb`
  - Solution: `refactored/<PART>/solutions/<SLUG>.ipynb`

## Pre-Work: MANDATORY Reading

1. `CLAUDE.md` - teaching conventions, tone, environment.
2. `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` - stack, datasets, models, chatbot through-line.
3. The `PLAN` file IN FULL - it specifies EXACTLY what each cell contains. Build it verbatim;
   do not invent content. The plan is the source of truth.
4. `initial_docs/outline.md` - this notebook's section (context only).

## Stack and conventions (THIS repo, NOT SageMaker/Barclays)

- **Environment: Google Colab.** Install cell: `!pip install -q ...` with pinned versions and
  `numpy<2`. Add the "restart runtime after install" note when gensim/numpy pins are present.
- Imports cell: standard libs + `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
  + `torch.manual_seed(42)`. NO SageMaker session. NO getpass. NO API keys. Everything runs from
  the HuggingFace Hub.
- Placeholders: exercise uses `None  # YOUR CODE` (never leaking the answer) + a verification block.
- Public datasets only (HF `datasets`: imdb, glue/sst2, glue/stsb, yelp_review_full, squad; or the
  course Dropbox CSVs named in the plan).
- Diagram placeholders: for each concept that the plan marks with a diagram, insert the EXACT
  comment `<!-- DIAGRAM: <description from plan> -->` followed by a link line:
  `[View diagram](../diagrams/<SLUG>/<diagram-slug>.mmd)`
  and 1-2 sentences on what it shows. The .mmd file does not exist yet; `/build-diagrams-auto`
  creates it later. Use the path `../diagrams/<SLUG>/<diagram-slug>.mmd` (relative to
  refactored/<PART>/exercises/). Record each diagram slug exactly as the plan names it.
- Tone: first person, friendly, heavy comments (what AND why). Theory -> Demo -> Lab; never more
  than 3 theory cells without a code cell.
- Plain ASCII only: no em dashes, no en dashes, no `---` separators inside cells, no Unicode
  multiplication sign, no emoji in code; emoji allowed only in markdown section headers if the
  plan uses them.

## Build procedure (autonomous)

1. Create the output dirs: `refactored/<PART>/exercises/` and `refactored/<PART>/solutions/`.
2. Create the empty exercise notebook with the Write tool (minimal nbformat 4.5 JSON skeleton,
   `"cells": []`).
3. Add cells ONE AT A TIME following the plan, using NotebookEdit with `edit_mode="insert"`:
   - First cell: no `cell_id`.
   - Every subsequent cell: pass `cell_id` = the id of the cell you are inserting AFTER, so order
     is deterministic. Before each insert after the first, read the notebook's current cell ids
     (small python snippet) and confirm position. Never trust append order.
   - For any single cell whose content exceeds ~30 lines, still insert it in one NotebookEdit call
     but do not batch multiple big cells into one tool call.
4. After every ~5 cells, run the structural check:
   `.venv/bin/python3 -m pytest`? No - just run:
   `python3 -c "import json,sys; json.load(open('<exercise path>'))"` to confirm valid JSON, and if
   `validate_notebooks.py` is present:
   `python3 validate_notebooks.py refactored/<PART>/exercises/<SLUG>.ipynb --type exercise`
   (use whatever python is available; prefer `.venv/bin/python3` if it exists, else `python3`).
   Do NOT stop for human approval - just fix any validation error and continue.
5. When the exercise notebook is complete, build the SOLUTION:
   - Copy the exercise to the solution path.
   - Walk each lab cell; replace `None  # YOUR CODE` with the complete implementation from the
     plan's solution content, adding explanation comments (what / why / common mistakes).
   - NEVER delete or reorder cells - only fill in placeholders, so the pair stays cell-aligned.
6. Final validation (best effort): if `validate_notebooks.py` exists, run
   `python3 validate_notebooks.py --pair refactored/<PART>/exercises/<SLUG>.ipynb refactored/<PART>/solutions/<SLUG>.ipynb`
   Confirm both files are valid JSON and under ~500 KB.

## AI-tells scan (MANDATORY before finishing)

Grep the two notebooks for em dashes, en dashes, and the Unicode multiplication sign; remove any
found. The course has a hard no-AI-tells rule.

## Report (your final message = the tool result)

Return a concise status: ID, exercise path, solution path, exercise cell count, solution cell
count, validation result (pass/fail + any errors), and the list of diagram slugs you referenced
(so the diagram builder knows what to produce).

## Notebook Edit Protocol

Follow the canonical safe procedure: normalize cell ids, locate cells by id AND content, control
insert position explicitly, read back and assert after each edit, run the structural gate
(valid JSON / nbformat) and a static check on concatenated code cells. No blind index rewrites.
