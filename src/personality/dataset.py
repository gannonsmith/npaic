import json
from torch.utils.data import Dataset
from src.util.build_prompt import build_prompt

class PersonalityDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                prompt = build_prompt(ex)
                target = ex["response"]

                full_text = prompt + " " + target
                self.data.append(full_text)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.data[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc