---
description: Validate exercise and/or solution notebooks for correctness
---

Validate notebooks: $ARGUMENTS

## Instructions

Run `validate_notebooks.py` (repo root) to verify:
- Python syntax correctness
- Import availability
- Exercise notebooks have proper `= None  # YOUR CODE` placeholders
- Solution notebooks have complete implementations (no `= None` in lab cells)
- Paired notebooks match structurally (same cell count + types)

## Usage Patterns

### Validate a single exercise notebook
```bash
python validate_notebooks.py 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb --type exercise
```

### Validate a single solution notebook
```bash
python validate_notebooks.py Solutions/1-Topic_Modelling_and_NER_Solution.ipynb --type solution
```

### Validate an exercise+solution pair together
```bash
python validate_notebooks.py --pair \
    2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb \
    Solutions/5-CBOW_Word_Embeddings_Solution.ipynb
```

### Validate all notebooks in a module
```bash
for nb in 1-Pre-NLP/*.ipynb; do
  python validate_notebooks.py "$nb" --type exercise
done
```

### Validate all solution notebooks
```bash
for nb in Solutions/*.ipynb; do
  python validate_notebooks.py "$nb" --type solution
done
```

## What to Check

✅ **Syntax passes** — No Python syntax errors in any code cell
✅ **Imports available** — All `import` statements resolve
✅ **Exercise has placeholders** — Lab cells contain `= None  # YOUR CODE`
✅ **Solution is complete** — No `= None` assignments in lab cells
✅ **Pair structure matches** — Same cell count and cell types
✅ **Cell order is sound** — Sections flow logically: Setup → Theory → Demo → Lab → Wrap-up; no forward references (a cell must not use a variable defined in a later cell)
✅ **Matches migration plan** — If `plans/migration_plan.md` exists, read the section for this notebook (search by notebook number/name) and verify: (1) section titles match the plan's outline, (2) all labs listed in the plan are present, (3) hyperparameter values match the plan's hyperparameter block, (4) the dataset URL matches the plan's dataset entry. Flag any deviation as a warning.

## Project Structure Reference

| Type | Location |
|------|----------|
| Module 1 exercises | `1-Pre-NLP/*.ipynb` |
| Module 2 exercises | `2-Text-Similarity/*.ipynb` |
| Module 3 exercises | `3-Text-Classification/*.ipynb` |
| Module 4 exercises | `4-Text-Generation/*.ipynb` |
| Framework exercises | `Frameworks/*.ipynb` |
| All solutions | `Solutions/*_Solution.ipynb` |

## When to Run

- After any agent builds a new notebook
- Before committing to git
- Before distributing to students
