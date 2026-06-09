**A1 NLP From Scratch notebooks – adversarial audit**

1. Runtime integrity  
- Exercise notebook errors out:  
  • Cell 20 loops `for doc in None:` → `TypeError: 'NoneType' object is not iterable`.  
  • Cell 25 later calls `sns.heatmap` after `crosstab = None` (will raise on attribute access).  
  • All OTHER sections guarded by “safety-net” run, but those two break a top-to-bottom run.  
- Undefined/late objects: every helper (`clean_text`, `nlp`, `bbc`) is defined before first use – good import order.  
- Solution notebook runs; every placeholder filled.

2. Pedagogical gaps  
- Sentence-transformer is first used in Cell 23 (“EMBED”) **before** students ever see a minimal embedding demo; concept appears suddenly.  
- `TfidfVectorizer` is referenced in Cell 14 comments but never imported or shown; students cannot relate the warning.  
- Topic “KMeans needs k” not explained until after students must choose `n_clusters` (they never choose – but comment says “fixed seed”): cognitive jump.

3. Lab authenticity & leakage  
- Labs are real (non-trivial code to write).  
- However Cell 14 “SAFETY-NET” *reveals a full working `reference_preprocess`* implementation; students can copy it.  
  Snippet:  
  ```python
  def reference_preprocess(text):
      cleaned = clean_text(text)
      doc = nlp(cleaned)
      return " ".join(
          token.lemma_
          for token in doc
          if token.is_alpha and not token.is_stop …
  ```  
- No other explicit answer leaks, but the above defeats Lab 2.1.

4. Solution-file hygiene  
- Solution notebook contains full answers (expected) but NOTHING is hidden/collapsible; every answer cell is immediately visible on open.  
- Exercise notebook’s safety-net cells are also fully visible (not collapsed metadata), leaking answers as noted.

5. Scriptability (`jupyter nbconvert --to python` then run)  
- Converted exercise script stops at the two `None` placeholders (same errors as #1) → will not finish.  
- Converted solution script would run end-to-end (no notebook-only magics).

6. Dependency pin  
- Both install cells correctly pin `numpy<2` (`# We pin numpy<2 because …`) – pass.

--------------------------------------------------
Ranked fixes
1. Replace explicit code in Cell 14 with a *hidden* `nbconvert "remove-input"` tag or at least collapse it; keep only a high-level hint.  
2. Add `# noqa`-style guard or separate **tests** instead of executing placeholder code so the notebook still runs when labs are empty.  
3. Insert a short demo cell **before** Cell 23 that shows embedding → vector shape and a 2-sentence rationale.  
4. Move KMeans parameter explanation *before* students see cluster results; add link to elbow/ silhouette criterion.  
5. Tag every solution cell in the *Solutions* notebook with the standard “solution” metadata and hide by default.  
6. Import `TfidfVectorizer` at first mention or remove the stray comment to avoid confusion.