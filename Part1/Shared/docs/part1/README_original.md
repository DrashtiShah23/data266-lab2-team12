# DATA 266 Lab 2 Part 1: Multimodal RAG

This implementation builds a read only multimodal retrieval system over the three supplied economics PDFs. It extracts page text, tables, figure metadata and page images, stores vectors in a persistent SQLite vector store, combines dense TF IDF plus latent semantic retrieval with BM25, and keeps conversation overrides outside the base index.

## Run locally

```bash
python run_public.py
python ../lab2/Fall\ 2026\ Lab\ 2/DATA266_Lab2_Part1/validate_submission.py submission.csv ../lab2/Fall\ 2026\ Lab\ 2/DATA266_Lab2_Part1/Questions.csv
```

The Kaggle notebook should set `QUESTIONS_PATH` when the hidden questions file is mounted, construct the index, and call `run_submission`. It must not contain question IDs, question text, or manually entered answer rows.

## Design choices for score

Exact figure and table names receive a retrieval boost so A-2 versus A-3 and A-4 versus A-5 are not mixed. Low trust Appendix C is indexed for discovery but is excluded from evidence. Revisions and outturns are represented in the policy answer layer. User supplied values are stored only in the conversation object and are never written to SQLite.

## Required evaluation artifacts

Before final tagging, add a notebook with the first cell printing commit, Python, package versions, GPU and seed. The notebook must compute baseline dense retrieval results, final hybrid retrieval results, Recall@5, citation accuracy, golden set correctness, two rater groundedness scores, agreement within one point, median latency, and the safety examples. Commit the 25 golden questions before tuning.

## Kaggle checklist

1. Upload the notebook with the three PDFs and manifest attached.
2. Run all cells on a GPU Kaggle session and save outputs.
3. Confirm the notebook writes `submission.csv` from the questions file.
4. Run the supplied validator.
5. Submit the output and keep the final notebook at the tagged `p1-v1.0` commit.
