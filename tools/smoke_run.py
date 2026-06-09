#!/usr/bin/env python3
"""Smoke-run a course notebook: convert to a script, shrink to tiny sizes, execute.

Usage:
    .venv/bin/python3 tools/smoke_run.py <notebook.ipynb>

Strategy (deliberately simple, per the "one batch of 5 sizes, nothing else" idea):
- Concatenate all CODE cells in order into one script.
- Replace common large knobs with tiny ones (subset=5, epochs=1, etc.) via regex.
- Neutralize things that block a headless smoke run: Gradio .launch(), pip installs,
  notebook magics, plt.show().
- Execute with a timeout; report PASS/FAIL + the first error and the failing line.

This is a SMOKE test (does the code path run), not a correctness or accuracy test.
"""
import json, re, sys, subprocess, tempfile, os, textwrap

def cells_to_script(path):
    nb = json.load(open(path))
    out = []
    for c in nb.get('cells', []):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        out.append(src)
    return '\n\n# ---- next cell ----\n'.join(out)

def shrink(script):
    s = script
    # strip notebook magics and shell lines, including backslash line-continuations
    cleaned = []
    skip_continuation = False
    for ln in s.splitlines():
        stripped = ln.lstrip()
        if skip_continuation:
            # previous shell/magic line continued onto this one
            skip_continuation = ln.rstrip().endswith('\\')
            cleaned.append('')
            continue
        if stripped.startswith('!') or stripped.startswith('%'):
            skip_continuation = ln.rstrip().endswith('\\')
            cleaned.append('')
            continue
        cleaned.append(ln)
    s = '\n'.join(cleaned)
    # tiny dataset subsets: any .select(range(N)) -> 5; train[:N] slices -> :5
    s = re.sub(r'range\(\s*\d{2,}\s*\)', 'range(5)', s)
    s = re.sub(r'(\[:\s*)\d{3,}(\s*\])', r'\g<1>5\g<2>', s)
    # common hyperparam knobs -> tiny
    s = re.sub(r'(num_train_epochs\s*=\s*)\d+', r'\g<1>1', s)
    s = re.sub(r'(NUM_EPOCHS\s*=\s*)\d+', r'\g<1>1', s)
    s = re.sub(r'(epochs\s*=\s*)\d+', r'\g<1>1', s)
    s = re.sub(r'(TRAIN_SUBSET\s*=\s*)\d+', r'\g<1>5', s)
    s = re.sub(r'(small_train\s*=\s*)\d+', r'\g<1>5', s)
    s = re.sub(r'(small_val\s*=\s*)\d+', r'\g<1>5', s)
    s = re.sub(r'(max_steps\s*=\s*)\d+', r'\g<1>1', s)
    # neutralize Gradio launches and blocking UI
    s = re.sub(r'\.launch\([^)]*\)', '.launch.__self__ if False else None', s)
    s = re.sub(r'\b\w+\.launch\b', 'None  # launch-neutralized', s)
    # matplotlib non-interactive
    s = "import matplotlib\nmatplotlib.use('Agg')\n" + s
    s = s.replace('plt.show()', 'plt.close("all")')
    return s

def main():
    if len(sys.argv) < 2:
        print('usage: smoke_run.py <notebook.ipynb>'); sys.exit(2)
    nb_path = sys.argv[1]
    script = shrink(cells_to_script(nb_path))
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(script); script_path = f.name
    env = dict(os.environ, TRANSFORMERS_OFFLINE='0', HF_HUB_DISABLE_TELEMETRY='1',
               TOKENIZERS_PARALLELISM='false')
    py = os.path.join(os.path.dirname(__file__), '..', '.venv', 'bin', 'python3')
    py = os.path.abspath(py)
    try:
        r = subprocess.run([py, script_path], capture_output=True, text=True,
                           timeout=int(os.environ.get('SMOKE_TIMEOUT', '600')), env=env)
    except subprocess.TimeoutExpired:
        print(f'RESULT: TIMEOUT\nscript: {script_path}'); sys.exit(1)
    ok = r.returncode == 0
    print(f'RESULT: {"PASS" if ok else "FAIL"}')
    print(f'script: {script_path}')
    if not ok:
        tail = (r.stderr or r.stdout).strip().splitlines()
        print('--- last 25 lines of error ---')
        print('\n'.join(tail[-25:]))
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
