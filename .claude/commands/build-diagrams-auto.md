---
description: AUTONOMOUS build of all Mermaid diagrams for one notebook from its plan, no human checkpoints. For use inside workflows/subagents. Writes .mmd files under refactored/<PART>/diagrams/<SLUG>/.
---

Build diagrams autonomously: $ARGUMENTS

(Argument: a notebook ID, its plan path, slug, and part folder, passed by the orchestrator.)

This is the AUTONOMOUS variant of `/build-diagrams`. There is NO human in the loop:
do NOT ask for approval between diagrams, do NOT design-in-chat-then-wait. Read the plan, build
every diagram file, fix any notebook links, then report.

## GUARD

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```
If missing, stop and report. Otherwise read it and keep it in mind.

## Inputs (provided by the orchestrator)

- `ID` - notebook id (e.g. A1)
- `PLAN` - `plans/refactor/notebooks/<ID>-<SLUG>.md`
- `SLUG` - e.g. `nlp-from-scratch`
- `PART` - `A`, `B`, or `C`
- Diagram output dir: `refactored/<PART>/diagrams/<SLUG>/`
- The built exercise notebook (for link verification):
  `refactored/<PART>/exercises/<SLUG>.ipynb`

## Step 1: Read context

1. Read the `PLAN` file. Find every diagram the plan calls for - any `<!-- DIAGRAM: ... -->`
   placeholder, any "Diagram slug:/Diagram path:/Description:" index entry, and any cell whose
   content references a diagram. Build the list of (diagram-slug, description, which cell/concept).
2. If the plan has no explicit diagram index but the built notebook has `<!-- DIAGRAM: ... -->`
   comments, derive the diagram list from those comments instead.
3. Read the relevant notebook cells so each diagram serves its exact concept.

## Step 2: Build each diagram (autonomous, no approval gates)

For each diagram:
1. `mkdir -p refactored/<PART>/diagrams/<SLUG>/`
2. Write `refactored/<PART>/diagrams/<SLUG>/<diagram-slug>.mmd` containing ONLY raw Mermaid source
   (no code fences, no surrounding prose).
3. Verify the notebook cell that references it links to the correct relative path
   `../diagrams/<SLUG>/<diagram-slug>.mmd` (relative to refactored/<PART>/exercises/). If the link
   is missing or wrong, fix it with a safe NotebookEdit (read the notebook, find the cell by id and
   content, edit, read back and assert).

## Diagram design rules

- One concept per diagram; do not draw a whole-system overview.
- `graph TD` for flows/pipelines; `sequenceDiagram` for request/response; `flowchart LR` for
  decision/routing.
- Max ~10 nodes; if more are needed, split into two diagrams.
- Node and edge labels: plain ASCII only. No em dashes, no en dashes, no Unicode multiplication,
  no emoji. Short phrases.
- Good fits for THIS course: the pipeline() task flow (text -> model -> output); encoder vs
  decoder vs encoder-decoder shapes; word2vec analogy vector arithmetic; the training loop
  (zero_grad -> forward -> loss -> backward -> step); embeddings-as-features (text -> doc vector ->
  MLP -> label); the chatbot path (user -> tokenizer -> fine-tuned model -> Gradio UI).

## Step 3: Validate

- Confirm every `.mmd` file is non-empty and starts with a valid Mermaid header
  (`graph`, `flowchart`, or `sequenceDiagram`).
- Confirm every `<!-- DIAGRAM -->` placeholder in the notebook now has a matching `.mmd` file and a
  resolvable link.

## Report (your final message = the tool result)

Return: ID, the diagram dir, the list of .mmd files written (with one-line descriptions), and
whether every notebook diagram link resolves (yes/no + which cells still need fixing).

## Notebook Edit Protocol

If you edit notebook cells to fix links: normalize cell ids, locate by id AND content, control
insert position, read back and assert after each edit. No blind index rewrites.
