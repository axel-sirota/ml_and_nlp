---
description: Build exercise and solution notebooks for a specific topic
---

Build notebooks for topic: $ARGUMENTS

**CRITICAL INSTRUCTIONS - READ CAREFULLY BEFORE STARTING**

## Command Arguments

This command takes TWO arguments:
1. **Topic name** (e.g., `topic_modelling`, `text_classification`, `cbow`)
2. **Environment** (one of: `colab`, `databricks`, `sagemaker`, `localhost`)

Example usage: `/build-notebook text_classification colab`

---

## Pre-Work: MANDATORY Reading & Planning

### Step 1: Read All Context Files (DO THIS FIRST)

**YOU MUST READ THESE FILES BEFORE DOING ANYTHING:**

1. **CLAUDE.md** - Project instructions and guidelines
2. **Existing notebooks in the repo** - Reference for structure and style

### Step 2: Plan the Notebook Structure

**IN CHAT, SHOW ME YOUR PLAN:**

List out:
1. Topic title and learning objectives
2. Real-world context/storytelling angle (CRITICAL - every topic needs a compelling story)
3. List of 2-4 main sections (break down the topic)
4. For EACH section, show:
   - Theory markdown (what concepts to cover)
   - Demo code (simple focused example)
   - Lab instructions (what students will build)
   - Starter code structure
5. Optional/Extra labs
6. Environment setup (based on environment argument)

**DO NOT PROCEED until I approve this plan.**

---

## Core Teaching Principles

### 1. Storytelling & Real-World Context
- **EVERY topic** starts with a real problem or business scenario
- Connect concepts to practical applications
- Frame public datasets in realistic contexts
- Example: "You're a data scientist at a bank processing thousands of checks daily..."

### 2. Heavy Documentation
- **Code cells**: Every line has meaningful comments (what AND why)
- **Markdown cells**: Detailed explanations with embedded code examples using ` ```python ``` ` blocks
- **Theory sections**: Clear but concise
- **Lab instructions**: Step-by-step, detailed enough for independent work

### 3. Demo-Driven Learning (Show, then Do)
- Demos are simple, focused examples
- Showcase ONE concept clearly
- Students see it done, then do it themselves in labs
- Pattern: **Theory -> Demo -> Lab**
- It is extremely important for every markdown cell we introduce some theory automatically next cell is code cell, even one line showing it in action. Dont chain more than 3 markdown cells of concepts without a code cell in the middle, at least showing, even if not real full demo.

### 4. Appropriate Difficulty
- Labs are **medium difficulty**: achievable in 15-30 minutes
- Focus on application and muscle memory
- Optional/extra labs provide challenge

### 5. Public Datasets Only
- MNIST, CIFAR-10, Iris, scikit-learn datasets, HuggingFace datasets
- Fetch from public URLs (torchvision.datasets, sklearn.datasets, etc.)
- Document dataset sources clearly

### 6. Tone: Friendly but Professional
- Conversational without being overly casual
- Encouraging and supportive
- Clear, direct instructions
- No jargon unless explained

---

## Notebook Structure Template

Each notebook MUST follow this structure:

### Header Section
1. **Cell 0 (Markdown)**: Topic title, learning objectives, prerequisites, session format, GPU setup (if Colab)
2. **Cell 1 (Markdown)**: "Section 0: Environment Setup" header
3. **Cell 2 (Code)**: Package installation (environment-specific)
4. **Cell 3 (Code)**: Import libraries, verify environment, set random seeds
5. **Cell 4 (Markdown)**: "What Are We Building Today?" - Real-world context/storytelling
6. **Cell 5 (Code)**: Preview of dataset or final outcome

### For Each Topic (2-4 topics per notebook)
1. **Markdown Cell**: Topic title + theory introduction
   - Real-world context paragraph
   - Concept explanation with inline code examples (use ` ```python ``` `)
   - "Demo: [Topic Name]" section
2. **Code Cell(s)**: Demo code (complete, heavily commented)
3. **Markdown Cell**: Lab instructions (detailed, step-by-step)
4. **Code Cell**: Lab starter code with `None # YOUR CODE` placeholders

### Closing Section
1. **Markdown Cell**: Optional/Extra labs (clearly marked)
2. **Code Cell(s)**: Optional lab starter code
3. **Markdown Cell**: Congratulations, what you learned, key takeaways, next steps, resources

---

## CRITICAL: Incremental Building - MANDATORY APPROVAL CHECKPOINTS

**YOU MUST NEVER ADD MORE THAN 5 CELLS WITHOUT EXPLICIT USER APPROVAL**

### MANDATORY Process (DO NOT DEVIATE):

1. **Add maximum 5 cells**
2. **STOP IMMEDIATELY**
3. **Ask user**: "I've added cells X-Y. How does it look? Should I continue?"
4. **DO NOT PROCEED** until user explicitly approves with words like "continue", "yes", "good", or "approved"
5. **If user says "continue"**: Return to step 1 for next batch of 5 cells ONLY
6. **NEVER interpret "continue" as permission to do ALL remaining cells**

### Violation Prevention:

**FORBIDDEN**: Adding 6+ cells without asking for approval
**FORBIDDEN**: Interpreting "continue" as "do everything"
**FORBIDDEN**: "Getting carried away with momentum"
**FORBIDDEN**: Assuming you can skip approval checkpoints

**REQUIRED**: Stop every 5 cells, no exceptions
**REQUIRED**: Wait for explicit approval each time
**REQUIRED**: Treat this as a hard checkpoint, not a suggestion
**REQUIRED**: After user says "continue", add ONLY the next 5 cells, then STOP again

### Why This Matters:
- Jupyter notebooks are JSON files that become HUGE very quickly
- User maintains control and can review quality at each stage
- Prevents file size issues and missing content
- Catches errors early before they propagate

### Implementation Example:

```
First batch (cells 0-4):
- Cell 0: Header markdown
- Cell 1: Environment setup markdown
- Cell 2: Package installation code
- Cell 3: Imports and verification code
- Cell 4: Real-world context markdown

STOP - Ask user: "I've added cells 0-4. How does it look? Should I continue?"
WAIT FOR APPROVAL

Second batch (cells 5-9):
- Cell 5: Dataset preview code
- Cell 6: Topic 1 theory markdown
- Cell 7: Topic 1 demo code
- Cell 8: Topic 1 lab instructions markdown
- Cell 9: Topic 1 lab starter code

STOP - Ask user: "I've added cells 5-9. How does it look? Should I continue?"
WAIT FOR APPROVAL

... repeat this cycle until notebook is complete
```

---

## Environment-Specific Setup

### Colab Environment

**Package Installation Cell:**
```python
# Install required packages (run this first in Google Colab)
# If running locally with conda/venv, you may skip this cell

!pip install [packages based on topic]
```

**Import Cell:**
```python
# Import all necessary libraries
import [libraries]

# Check versions
print(f"Library version: {library.__version__}")

# Device configuration - automatically use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Set random seed for reproducibility
torch.manual_seed(42)

print("\n Environment setup complete!")
```

### Databricks Environment

**Package Installation Cell:**
```python
# Install additional packages not in Databricks ML Runtime
%pip install [additional packages]
```

### SageMaker Environment

**Package Installation Cell:**
```python
# Install required packages
!pip install [packages]
```

### Localhost Environment

**Package Installation Cell:**
```python
# Ensure you're in your virtual environment
# Run: python -m venv academy-env
# Activate: source academy-env/bin/activate (macOS/Linux)
#        or academy-env\\Scripts\\activate (Windows)

# Install packages (run in terminal):
# pip install [packages]

# This cell is informational - run the pip commands in your terminal
```

---

## Exercise vs Solution Notebooks

### Exercise Notebook

**Lab starter cells should have:**
```python
# Your code here

# 1. [Task description]
variable_1 = None  # YOUR CODE

# 2. [Task description]
variable_2 = None  # YOUR CODE

# Verification (provide this)
if variable_1 is not None:
    print(f"Result: {variable_1}")
```

**Characteristics:**
- Full theory and explanations
- Complete demo code
- Detailed lab instructions
- Starter code with `None # YOUR CODE` placeholders
- Verification/test code (so students know if they're right)

### Solution Notebook

**Lab solution cells should have:**
```python
# Solution: Lab X.Y - [Lab Title]

# 1. [Task description]
variable_1 = actual_implementation  # Complete implementation with comment

# 2. [Task description]
variable_2 = actual_implementation  # Complete implementation with comment

# Verification
if variable_1 is not None:
    print(f"Result: {variable_1}")

# Explanation:
# [Detailed explanation of the solution approach]
# [Common mistakes students make]
# [Alternative approaches]
```

**Characteristics:**
- Everything from exercise notebook
- Fully completed lab code
- Extensive explanatory comments
- Expected outputs visible (can include output in cells)
- Notes on common mistakes or alternatives

---

## CRITICAL: NotebookEdit Best Practices

**ALWAYS INSERT CELLS AFTER A SPECIFIC CELL, NEVER AT THE TOP!**

### WRONG - Will insert at top of notebook:
```python
NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    cell_type="markdown",
    edit_mode="insert",
    new_source="content"
)
```

### CORRECT - Insert after specific cell:
```python
NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    cell_id="cell-5",  # CRITICAL: Specify the cell to insert AFTER
    cell_type="markdown",
    edit_mode="insert",
    new_source="content"
)
```

### Workflow for Building Notebooks:

1. **Create empty notebook** with Write tool (basic JSON structure)
2. **Add first cell** with `edit_mode="insert"` (no cell_id needed for first cell)
3. **Add subsequent cells** ALWAYS using `cell_id` parameter:
   ```python
   # Add cell 1
   NotebookEdit(..., cell_id="cell-0", ...)  # Insert after cell-0

   # Add cell 2
   NotebookEdit(..., cell_id="cell-1", ...)  # Insert after cell-1

   # Add cell 3
   NotebookEdit(..., cell_id="cell-2", ...)  # Insert after cell-2
   ```

4. **Track cell IDs**: Note the returned cell_id from each NotebookEdit call and use it for the next insertion

### Common Mistake:
- **Do NOT** forget the `cell_id` parameter after the first cell
- **Do NOT** assume cells will be inserted at the end automatically
- **Always specify** which cell to insert after

---

## Workflow Summary

1. **Read all context files** (mandatory)
2. **Plan notebook structure** and get my approval
3. **Build EXERCISE notebook incrementally:**
   - Create skeleton
   - Add 5 cells at a time
   - After each batch, STOP and ask: "I've added cells X-Y. How does it look? Should I continue?"
   - Wait for approval
   - Continue until complete
4. **Build SOLUTION notebook incrementally:**
   - Copy structure from exercise
   - Fill in solutions
   - Add explanatory comments
   - Add 5 cells at a time with approval
5. **Final verification:**
   - Both notebooks run top-to-bottom without errors
   - Markdown renders correctly
   - File sizes are reasonable (<500KB)

---

## Checklist (Use this for every notebook)

Before marking a notebook as complete:

- [ ] Topic title and learning objectives clear
- [ ] Environment setup instructions included (environment-specific)
- [ ] Real-world context/storytelling for the topic
- [ ] Each section has: theory -> demo -> lab structure
- [ ] All demo code is heavily commented
- [ ] Lab instructions are detailed and step-by-step
- [ ] Code examples in markdown use ` ```python ``` ` blocks
- [ ] Public datasets only, with clear documentation
- [ ] Starter code uses `None # YOUR CODE` pattern
- [ ] Solution notebook has complete code with extensive comments
- [ ] Optional/extra lab included at end
- [ ] Markdown renders correctly (verified)
- [ ] Notebook runs top-to-bottom without errors
- [ ] Tone is friendly but professional
- [ ] All labs build toward topic's main outcome
- [ ] File size is reasonable

---

**NOW BEGIN:**

1. Read all context files
2. Show me the plan for topic [NAME] with environment [ENVIRONMENT]
3. Wait for my approval
4. Start building incrementally (5 cells at a time)
5. **REMEMBER**: Always use `cell_id` parameter to insert cells in correct order!
