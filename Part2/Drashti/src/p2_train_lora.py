"""DATA 266 Part 2 data preparation and LoRA training.

The script loads the captioned Hugging Face dataset, removes byte-identical
duplicates, creates the required seed 266 split, and trains one UNet LoRA run.
Run it twice with different ranks and the same step budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATASET_ID = "lambdalabs/naruto-blip-captions"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"


def image_hash(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def prepare_dataset(output_dir: Path):
    ds = load_dataset(DATASET_ID, split="train")
    seen = set()
    keep = []
    removed = 0
    for i, row in enumerate(ds):
        try:
            image = row["image"].convert("RGB")
            key = image_hash(image)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            keep.append(i)
        except Exception:
            removed += 1

    clean = ds.select(keep).shuffle(seed=266)
    split = clean.train_test_split(test_size=0.10, seed=266)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": DATASET_ID,
        "base_model": BASE_MODEL,
        "original_rows": len(ds),
        "removed_corrupt_or_duplicate": removed,
        "clean_rows": len(clean),
        "train_rows": len(split["train"]),
        "test_rows": len(split["test"]),
        "split_seed": 266,
        "limitations": [
            "The captions are short and can omit visual details.",
            "The dataset represents a narrow anime character domain and may not generalize to photographs or other art styles.",
        ],
    }
    (output_dir / "data_split.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return split, metadata


class CaptionImageDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = row["image"].convert("RGB")
        tokens = self.tokenizer(
            row["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.transform(image), "input_ids": tokens}


def train(rank: int, steps: int, output_dir: Path, batch_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Part 2 training")

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    split, metadata = prepare_dataset(output_dir.parent / "data")
    device = torch.device("cuda")
    dtype = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device, dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet.add_adapter(lora_config)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    loader = DataLoader(CaptionImageDataset(split["train"], tokenizer), batch_size=batch_size, shuffle=True, num_workers=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    unet.train()
    step = 0
    while step < steps:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with torch.autocast(device_type="cuda", dtype=dtype):
                prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == steps:
                print({"rank": rank, "step": step, "steps": steps, "loss": round(float(loss), 5)})
            if step >= steps:
                break

    unet.save_attn_procs(str(output_dir))
    run = {
        **metadata,
        "rank": rank,
        "learning_rate": 1e-4,
        "steps": steps,
        "batch_size": batch_size,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "training_seconds": round(time.perf_counter() - start, 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "device": torch.cuda.get_device_name(0),
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.rank, args.steps, args.output, args.batch_size)


if __name__ == "__main__":
    main()
