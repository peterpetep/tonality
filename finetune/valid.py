import torch
import random
from transformers import AutoTokenizer
from datasets import DatasetDict
from finetune.regression_model import RegressionModel
from safetensors.torch import load_file

from hotfixtokenizer import hotfixtokenizer


def predict_with_chunks(model, tokenizer, lyrics, device, max_length=512):
    tokens = tokenizer(lyrics, truncation=False, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]

    content_ids = input_ids[1:-1]
    content_mask = attention_mask[1:-1]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    content_length = max_length - 2 #510

    chunks_ids, chunks_mask = [], []
    for start in range(0, len(content_ids), content_length):
        chunk = content_ids[start : start + content_length]
        mask = content_mask[start : start + content_length]

        pad_len = content_length - len(chunk)
        chunk = torch.cat([
            torch.tensor([cls_id]), 
            chunk, 
            torch.tensor([sep_id]), 
            torch.zeros(pad_len, dtype=torch.long)
        ])
        mask = torch.cat([
            torch.ones(1, dtype=torch.long), 
            mask, 
            torch.ones(1, dtype=torch.long), 
            torch.zeros(pad_len, dtype=torch.long)
        ])

        chunks_ids.append(chunk)
        chunks_mask.append(mask)

    batch_ids = torch.stack(chunks_ids).to(device)
    batch_mask = torch.stack(chunks_mask).to(device)

    with torch.no_grad():
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        preds = outputs['logits'].detach().cpu()

    content_lengths = torch.tensor([mask.sum().item() - 2 for mask in chunks_mask])  # subtract CLS/SEP
    weights = content_lengths.float() / content_lengths.sum()
    return (preds * weights.unsqueeze(1)).sum(dim=0)


model_path = "./deberta-MERGE-tuned-prod"
dataset_path = "finetune/MERGE_PREPPED"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)

hotfixtokenizer(tokenizer)

model = RegressionModel("microsoft/deberta-v3-base")
weights = load_file(f"{model_path}/model.safetensors")
model.load_state_dict(weights)
model.to(torch.float32)
model.to(device)
model.eval()

dataset = DatasetDict.load_from_disk(dataset_path)
val_split = dataset['validation']

arousal_errors = []
valence_errors = []

for sample in val_split: 

    with torch.no_grad():
        
        preds = predict_with_chunks(model, tokenizer, sample["Lyrics"], device)
        pred_arousal, pred_valence = preds[0].item(), preds[1].item()

    arousal_errors.append(abs(pred_arousal - sample["Arousal"]))
    valence_errors.append(abs(pred_valence - sample["Valence"]))


print(f"{'Metric':<10} | {'MAE':<10}")
print(f"{'Arousal':<10} | {sum(arousal_errors)/len(arousal_errors):<10.4f}")
print(f"{'Valence':<10} | {sum(valence_errors)/len(valence_errors):<10.4f}")