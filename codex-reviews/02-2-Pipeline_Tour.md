Notebook audited:  
exercise file `exercises/1-Pre-NLP-Tools/2-Pipeline_Tour.ipynb` (31 cells)  
solution file `Solutions/1-Pre-NLP-Tools/2-Pipeline_Tour.ipynb` (31 cells)

==================================================================
1. Run-time / ordering problems
- Cell 2 (`!pip install …`) is a shell-magic that crashes if the NB is executed as a plain Python script.  
- All later cells rely on variables created earlier (`device`→3, `classifier`→10, `ticket`→6, `router`→13) and the order is consistent; no undefined-name errors if run top-to-bottom.  
- Gradio demo (cell 28) is wrapped in `try/except ImportError`, so missing dependency is handled.  
- Labs leave `None` placeholders; verification prints but never raises, so NB still “runs”.

2. Pedagogical gaps
- Sentiment pipeline (cell 10) is used before the concept of model swapping is explained; the jump is small and probably OK.  
- No explanation of what “zero-shot” means **before** students see the API call in cell 13.  
- Device mapping (`device = 0/-1`, cell 3) is described but GPU vs CPU speed impact is never demoed.  

3. Lab quality & leakage
- Both labs are genuine implementation tasks (`my_labels = None`, `twitter_sentiment = None`, etc.).  
- Starter comments *do not* leak the final answer; they only give call signatures. Good.

4. Solution visibility / safety-nets
- Solution notebook has fully filled answers in plain sight:  
  • Cell 15 header `# Lab 1: Zero-shot ticket router. SOLUTION.`  
  • Cell 27 header `# Lab 2: Swap a model and compare on the same task. SOLUTION.`  
  No metadata (`{"tags":["solution"]}`) or collapsible blocks are used, so answers are shown immediately.

5. Scriptability
- Because of the `!pip` call (cell 2) and magic comments, the notebook cannot be run with `python 2-Pipeline_Tour.py` without manual edits. Everything else is pure Python.

6. Dependency pin
- The install cell explicitly pins `numpy<2` (`"numpy<2"` in cell 2) – correct.

==================================================================
Concrete fixes (priority order)

1. Add `{"tags":["solution"]}` + “hide input” metadata or Jupyter **collapsible/Details** blocks to every solution cell so students do not see answers by default.  
2. Provide a requirements file or wrap the `!pip` install in `if __name__ == "__main__":` guard so the notebook can be converted to a script without errors.  
3. Insert a 1-sentence definition of “zero-shot classification” *before* cell 13 to avoid the unexplained jump.  
4. Add a quick timing demo (CPU vs GPU) right after device selection to motivate the `device` logic.  
5. Make the lab verification assert on `None` placeholders (or raise `NotImplementedError`) so failing to implement actually fails; this prevents silent passes.