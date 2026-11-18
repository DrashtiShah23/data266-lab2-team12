# AI usage

| Tool | Version | Used for |
|---|---|---|
| Claude (Anthropic), claude.ai | Claude Opus 5.5 | Generated the notebook code for Parts 1–3, the scenario/red-team files, the golden-set draft (verified by us against the PDFs), debugging help and a draft of the report. We ran everything, reviewed all code and edited the report. |
| _add any others_ | | |

All evaluation data (golden set questions, human ratings), the report text and all reported numbers were produced by the team; every number in the report comes from a notebook output.

## AI-assisted judging (instead of human ratings)
We did not complete the human ratings in time. The per-item scores in `eval/p*_ai_judge.csv` files in each part folder (15 Part 1 answers, 8 Part 2
image comparisons, 15 Part 3 results) were produced by Claude (Anthropic) from the saved notebook outputs, at our request.
They are labelled as AI-judge results in the report and are not presented as human ratings.
