Audit of exercises/3-Transfer-Chatbot/10-EncoderDecoder_QA_Chatbot.ipynb and its solution.

1. Runtime correctness
   - Cell 2: installs pin `numpy<2` (good) but omits `torch`, so a fresh Colab runtime errors on the *import* cell that immediately follows.
   - Import/seed cell runs, but later cells break:
     • Cell ~219 uses `AutoModelForSeq2SeqLM.from_pretrained` before `pip install` finishes if the user skips the requested restart (common in “Run-all”).  
     • Demo `_demo_args = Seq2SeqTrainingArguments(..., eval_strategy="no")` (Cell ~540) and Lab 2 `training_args = Seq2SeqTrainingArguments(..., eval_strategy="epoch")` use the non-existent keyword `eval_strategy` instead of `evaluation_strategy` (HF 4.57.1). Raises `TypeError`.
     • Lab 1 placeholder function returns `None`; the immediate verification cell asserts and halts execution *before* the safety-net cell can patch it.
   - Therefore the notebook cannot run top-to-bottom without manual fixes even if the student skips the labs.

2. Pedagogical gaps
   - `Seq2SeqTrainingArguments` fields (`predict_with_generate`, lr choice, fp16) are demanded in Lab 2 **before** those hyper-parameters are explained in prose; the “theory cell” talks about them only implicitly (“the flag that makes it generate while evaluating”), leaving beginners guessing values.
   - The need to call `.to(device)` on the *tokenized* tensors is never discussed; context switches between CPU/GPU silently.
   - The data-collator’s role in padding `labels` to `-100` is asserted (“classic beginner mistakes”) but never *shown* in a demo.

3. Lab quality and leakage
   - Labs are genuine coding tasks (`preprocess`, `Seq2SeqTrainingArguments`), so ✅ “real”.
   - However each lab’s safety-net **contains the full reference answer in clear text**:
     • Cell ~460: `preprocess` full implementation.  
     • Cell ~680: complete `training_args` + call to `trainer.train()`.  
     This leaks the solution to any student who scrolls or searches.

4. Solution notebook hygiene
   - The solution notebook has every answer already filled; cells are not collapsed or tagged “solution”. Nothing hides them; opening the file shows code immediately.

5. Scriptability
   - Because of (a) the bad `eval_strategy` keyword and (b) the failing Lab 1 asserts, `papermill`/`pytest` style automation would stop with an exception long before the end. Even if asserts were removed, the undefined-lab placeholders would propagate `None` into `.map()` and crash.

6. numpy pin
   - The install cell does include `numpy<2` ✅.

----------------------------------------------------
Concrete fixes (highest impact first)
1. Replace every `eval_strategy` with the correct `evaluation_strategy` and test on transformers 4.57.1.
2. Move the Lab 1 verification *after* the safety-net (or drop the assert) so “Run-all” continues even when the lab is blank.
3. Convert safety-net answers into *hidden* collapsible cells (`{"tags":["solution"]}`) or external utils; never expose full code inline.
4. Add explicit exposition/demonstration of:
   - why `predict_with_generate=True` is required,
   - how the data-collator turns pad tokens to `-100`.
5. Add `torch==2.2.*` (or course pin) to the install cell to guarantee imports on a clean Colab.
6. In the solution notebook, tag all answer cells with “solution” and set `metadata.hidden=true` so students don’t see them by default.