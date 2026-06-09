Audit of `exercises/2-Build-It/8-Capstone_B.ipynb` (student) vs `solutions/2-Build-It/8-Capstone_B.ipynb` (staff) — cell numbers are zero-based JSON indices.

1. Correct execution  
- All imports are present and ordered; `wv`, `doc_vector`, `make_loader`, etc. are defined before use.  
- Safety-net cells (14, 22, 28, 31) guarantee downstream cells run even when labs are skipped.  
- Fatal script errors found: none inside notebook, but see criterion 5.  

2. Pedagogical gaps  
- Cell 6 loads IMDB with no reminder of how `datasets.load_dataset` works (last shown in A-track notebooks).  
- Logistic-regression baseline (16) appears without any prior demonstration of logistic regression in the Build-It track.  
- `train_test_split` is imported (3) but never used, hinting at a missing demonstration of validation-set carving.  

3. Lab quality & leaking hints  
Real labs are present:  
  • Lab 1 doc-vector averaging (13)  
  • Lab 2 MLP class (21)  
  • Lab 3 training loop (27)  
  • Lab 4 test-set evaluation (30)  
Tasks are blanked with `YOUR CODE`.  
Leakage: every safety-net cell shows the exact solution logic. Example (cell 22):  
    class MLPClassifier(nn.Module):  
        …  
        self.net = nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Linear(hidden,n_classes))  

4. Solution notebook hygiene  
The staff notebook has every answer filled in and **none of the answer cells are hidden or collapsible** (`metadata` is `{}`), so the full solution is visible on first open.  

5. Plain-script runability  
Fails: cell 2 starts with `!pip install …`, which is Jupyter-only. Exclamation marks will raise a `SyntaxError` in a `.py` script.  

6. Pinned NumPy < 2  
Yes. Cell 2 explicitly installs `"numpy<2"`.

----------------------------------------------------------------
Ranked list of concrete fixes
1. Tag safety-net/answer cells with `{"tags":["solution"]}` and `{"collapsed":true}` (or Colab’s “hide-code” metadata) in **both** notebooks.  
2. Replace `!pip install …` with a guarded Python call (`subprocess` or a commented shell line) so the file can run as a script.  
3. Move a short refresher on `load_dataset` and logistic regression *before* they are first used, or link back to earlier notebooks.  
4. Delete the unused `train_test_split` import or demonstrate its use to create the validation split.  
5. In the student notebook, keep safety-nets but wrap them in `if False:` or `%load` links so students do not read the answer accidentally.  
6. Add a one-sentence reminder that averaging word vectors was first covered in B5 to reinforce continuity.