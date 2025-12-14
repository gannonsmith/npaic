import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.amp import autocast
import wandb
from tqdm import tqdm
from pathlib import Path

from src.personality.dataset import PersonalityDataset
from src.personality.lora_setup import setup_lora

BASE_MODEL_PATH = "models/base/qwen2.5-3b"
OUTPUT_DIR = "models/lora_adapters/arthur_morgan"

TRAIN_FILE = "data/summarized_splits/dialogue_pairs_train_summarized.jsonl"
VAL_FILE = "data/summarized_splits/dialogue_pairs_val_summarized.jsonl"

def train(
        num_epochs=3,
        grad_accum=4,
        log_every=50,
        save_every=500,
        batch_size=4,
        lr=2e-4
    ):
    wandb.init(
        project="npaic-personality",
        name="arthur-lora-qwen2.5-3b"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model = setup_lora(model)
    model.print_trainable_parameters()

    train_ds = PersonalityDataset(TRAIN_FILE, tokenizer)
    val_ds = PersonalityDataset(VAL_FILE, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=True
    )

    step = 0
    model.train()

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step * grad_accum + 1) % log_every == 0:
                wandb.log({"train_loss": loss.item() * grad_accum})

            if (step * grad_accum + 1) % save_every == 0:
                save_path = f"{OUTPUT_DIR}/checkpoint_{epoch}_{step}"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_path)
            
            step += 1
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
            
        val_loss /= len(val_loader)
        wandb.log({"val_loss": val_loss})
        model.train()
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    wandb.finish()

if __name__ == "__main__":
    train()