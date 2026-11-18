# DATA 266 Lab 2

Repository for all three lab parts completed by the team.

## Repository structure

```text
data/
  part1/              Supplied Part 1 knowledge base and public questions
src/
  part1/              Part 1 retrieval and answering implementation
Part1/Shared/notebooks/
  p1_rag.ipynb        Executed Part 1 notebook
  p2_finetune.ipynb   Add during Part 2
  p3_agents.ipynb     Add during Part 3
outputs/
  part1/              Public and final generated submissions and artifacts
docs/
  part1/              Golden set and Part 1 notes
report/               Final IEEE report
models/               Links and cards for Hugging Face artifacts
part2/                Part 2 source files and figures
part3/                Part 3 source files and demo assets
```

## Run Part 1 locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 Part1/Shared/run_p1.py
python3 Part1/Shared/data/part1/validate_submission.py Part1/Shared/outputs/part1/submission_public.csv Part1/Shared/data/part1/Questions.csv
```

The public questions are for practice. The final Kaggle notebook must read the mounted questions path and generate `submission.csv` without manually entering answers.

The retrieval code uses `sentence-transformers/all-MiniLM-L6-v2` when that model is available, blended with TF IDF and BM25. It automatically falls back to the lexical index if the model cannot be loaded. For a fast local smoke test, set `USE_SEMANTIC_EMBEDDINGS=0`.

## GitHub workflow

Create one private repository named `data266-lab2-team##`, open this folder in Cursor, commit the initial Part 1 work, and push it to GitHub. Later add Part 2 and Part 3 to this same repository. Before submission create the required tags `p1-v1.0`, `p2-v1.0`, and `p3-v1.0`.

## Current status

Part 1 implementation and public submission are present. Part 1 evaluation tables, the executed notebook outputs, final report, Part 2, and Part 3 still need to be completed before final submission.
