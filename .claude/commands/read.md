---
description: Read all project context files before starting any work
---

Read the following files to load full project context:

1. `CLAUDE.md` - teaching philosophy, notebook structure, environment, critical rules
2. `initial_docs/outline.pdf` - full course outline (9 topics, 3 days)
3. `initial_docs/technical_setup.md` - Barclays infrastructure spec (SageMaker, Ubuntu VM, S3 buckets)

Then report back:
- Which topics have plans in `plans/`
- Which topics have exercises in `exercises/`
- Which topics have solutions in `solutions/`
- What is the next topic to work on

---

## Notebook Edit Protocol (awareness)

If this skill ends up editing notebook cells (not just reading them), follow
the canonical procedure in `~/.claude/NOTEBOOK_EDIT_PROTOCOL.md`: normalize
cell ids, size-gate the mechanism, locate cells by id + content, read back and
assert after every edit, and run the structural + static code gates. Blind
bulk index-based rewrites are forbidden.
