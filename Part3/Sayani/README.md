# DATA 266 Lab 2 — Part 3: Five-agent multimodal system (Sayani)

Pair_Programming_Team_12. Executed notebook: `notebooks/p3_agents.ipynb` (outputs saved, ends with the Gradio app).

All models are open-weight and run locally on one Kaggle Tesla T4 (16 GB). **No hosted APIs are used.**
Seed 266. Exact library versions: `requirements.txt` in this folder. Weights and images live on Hugging Face, never in Git.

## Agents
Planner (input validation, safety, routing) → Vision (Qwen2.5-VL-3B) → Prompt engineer → Generation
(SD 1.5 img2img, InstructPix2Pix, SD 1.5 + our Part 2 LoRA r32) → Critique (CLIP score, style shift, VLM judgement,
safety; at most 2 revisions). One shared typed state; every step is logged to a JSON trace.

Results: task success 13/15 without critique → 15/15 with critique (automatic); red-team attack success 0/10.

## How to run (Kaggle)
Accelerator **GPU T4 x1**, Internet **On**. Attach a dataset with `eval/p3_scenarios.json` and `eval/p3_redteam.json`.
Run All. The last cell launches the Gradio demo with a public share link.

## Files
- `notebooks/p3_agents.ipynb` — executed notebook
- `eval/p3_scenarios.json`, `eval/p3_redteam.json` — 15 scenarios and 10 red-team cases
- `eval/p3_ai_judge.csv` — AI-assisted pass/fail judgements (not human ratings; see AI_USAGE.md)
