# DATA 266 Lab 2 — Part 2: Stable Diffusion 1.5 LoRA fine-tuning (Sayani)

Pair_Programming_Team_12. Executed notebook: `notebooks/p2_finetune.ipynb` (outputs saved).

All models are open-weight and run locally on one Kaggle Tesla T4 (16 GB). **No hosted APIs are used.**
Seed 266. Exact library versions: `requirements.txt` in this folder. Weights and images live on Hugging Face, never in Git.

## What it does
Dataset `lambdalabs/naruto-blip-captions` (1,221 images, 90/10 split). VAE latents and CLIP text embeddings are
computed once; two LoRA runs on the UNet attention layers differ only in rank (r=4, r=32): lr 1e-4 cosine, 400 steps,
batch 4, fp16. Evaluation on 20 fixed prompts × 2 seeds: KID (primary), CLIP score, LPIPS diversity, memorisation
check, NSFW safety checker. Final model: rank 32.

| | Base | LoRA r=4 | LoRA r=32 |
|---|---|---|---|
| KID in-domain ↓ | 0.1049 | 0.0613 | **0.0531** |
| CLIP score | **32.52** | 31.85 | 31.67 |

## How to run (Kaggle)
Accelerator **GPU T4 x1**, Internet **On**, secret `HF_TOKEN` (write). Attach a dataset with `eval/p2_prompts.json`.
Save Version → Save & Run All (about 1 hour). Adapters, images and the model card are pushed to Hugging Face
(see `links.txt`).

## Files
- `notebooks/p2_finetune.ipynb` — executed notebook
- `eval/p2_prompts.json` — 20 test prompts (fixed before evaluation)
- `eval/p2_ai_judge.csv` — AI-assisted side-by-side judgements (not human ratings; see AI_USAGE.md)
