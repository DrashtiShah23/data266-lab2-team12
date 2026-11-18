# DATA 266 Lab 2 — Part 1: Multimodal RAG (Sayani)

Pair_Programming_Team_12. Executed notebook: `notebooks/p1_rag.ipynb` (outputs saved). Kaggle public score: 0.40.

All models are open-weight and run locally on one Kaggle Tesla T4 (16 GB). **No hosted APIs are used.**
Seed 266. Exact library versions: `requirements.txt` in this folder. Weights and images live on Hugging Face, never in Git.

## Pipeline
PyMuPDF text chunks + figure/table crops (exact PDF text kept per region) → Qwen2.5-VL-3B figure descriptions →
`BAAI/bge-small-en-v1.5` embeddings in ChromaDB + BM25 → reciprocal-rank fusion, exact-figure pinning,
`BAAI/bge-reranker-base` (bonus) → Appendix C (untrusted) filtered out → Qwen2.5-VL-3B JSON answer with status,
figure references and citations; per-conversation memory for user-supplied values.

## How to run (Kaggle)
1. Accelerator **GPU T4 x1**, Internet **On**. Attach the competition data and a dataset containing `eval/golden_set.csv`.
2. Run All. The notebook builds the index, evaluates the golden set (baseline / hybrid / final), answers the 40
   questions, runs `validate_submission.py` and writes `submission.csv`. Set `QUESTIONS_PATH` for a hidden file.

## Files
- `notebooks/p1_rag.ipynb` — executed notebook
- `eval/golden_set.csv` — 25-item golden set
- `eval/p1_ai_judge.csv` — AI-assisted ratings of 15 answers (not human ratings; see AI_USAGE.md)
- `results/submission.csv` — the Kaggle submission produced by the notebook

## Golden-set note
`eval/golden_set.csv` is the file the notebook was evaluated on. Three items (G19, G21, G22) share wording with
practice questions; no code reads question text or IDs to choose answers. The report gives the score with them (9/25)
and without them (8/22). The golden set was committed after tuning; the Git history shows this.
