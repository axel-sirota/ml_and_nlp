Notebook audited:  
- exercises/3-Transfer-Chatbot/11-Capstone_C.ipynb (“student”)  
- Solutions/3-Transfer-Chatbot/11-Capstone_C.ipynb (“solution”)

───────── 1. Runtime integrity ─────────
A. Student
- Cell 5 imports both encoder-only and encoder-decoder classes in one shot; OK.
- Cell 7 leaves `PATH = None`; every later cell gates on it → assertion in same cell stops execution until student edits (intentional).
- Each lab cell ends with hard `assert`s that will raise if TODOs remain (`tokenizer is not None`, etc.). Good for gating but means sequential run fails out-of-the-box.
- Safety-net cells (16, 24) call objects built in previous labs; if earlier cells were skipped they still rely on variables declared there (`tokenizer`, `TEXT_FIELD`, etc.) – but they re-create them, so execution continues. No undefined names detected.
- Colab restart requirement after pip-install is noted; running as a script without restart keeps the old NumPy 2.x → import of torch-compiled wheels with numpy<2 assumptions will crash.

B. Solution
- All TODOs are filled so no assert fails.
- Same Colab-restart caveat.

───────── 2. Pedagogical gaps ─────────
- Lab A/B ask learners to build `Trainer`/`Seq2SeqTrainer` objects but the notebook never re-shows the minimal pattern (last seen in previous chapter C9/C10). A quick recap code block is missing before Lab A3/B2.
- The need to rename “label”→“labels” is mentioned but not demonstrated; first-time learners may not know the `.rename_column` API.
- Safety-net cells silently create an UNTRAINED model; learners could mis-interpret later metrics unless the teacher explains this upfront (only a brief comment is present).

───────── 3. Lab authenticity & leakage ─────────
Labs are genuine implementation tasks (placeholders set to `None`).  
No direct variable answers in the student file, but two comment leaks:
- Cell 12 comment: “honest held-out split for it is validation” gives away which split to choose.
- Cell 19 comment: “t5-small is tiny, fine-tuning takes ~8-10 min on a T4” reveals the exact model expected.

───────── 4. Solution visibility / safety-nets ─────────
- Solution notebook contains fully filled code (expected).  
- In the student notebook both Path-A and Path-B safety-net cells are completely visible; `metadata` has no `"collapsed": true` or `"source_hidden": true`, so answers are one scroll away. They should be hidden behind a collapsed cell or moved to a separate helper file.

───────── 5. Plain-script executability ─────────
- Fails: requires manual PATH choice, Colab restart after pip install, and interactive Gradio launch. Without notebook UI the flow stops after the pip cell or gets stuck awaiting user input.

───────── 6. numpy<2 pin ─────────
- Both notebooks’ first cell installs `"numpy<2"` (line 43 in each file) ✔

───────── Ranked fixes ─────────
1. Mark both safety-net cells’ metadata `{"jupyter":{"source_hidden":true,"collapsed":true}}` so students don’t see fallback code first.  
2. Add a recap code snippet (or link) before Lab A3/B2 showing a minimal `TrainingArguments`/`Trainer` build.  
3. Move Colab-specific restart instructions to a markdown note and guard the pip install with `if 'google.colab' in sys.modules:` to make script mode survivable.  
4. Remove answer-hinting comments (“validation split”, timing estimates) or re-phrase to avoid giving away decisions.  
5. Add a one-line default `PATH = "A"` so running top-to-bottom in script mode works (still gated by asserts).  
6. In safety-net cells, create any required objects completely inside the block to avoid hidden dependencies on unfinished labs.