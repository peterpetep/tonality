import torch
import random
from transformers import AutoTokenizer
from datasets import DatasetDict
from finetune.regression_model import RegressionModel
from safetensors.torch import load_file


def predict_with_chunks(model, tokenizer, lyrics, device, max_length=512, overlap=50):
    
    tokens = tokenizer(
        lyrics,
        truncation=False,
        return_tensors="pt"
    )
    
    input_ids = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]
    
    
    if len(input_ids) <= max_length:
        inputs = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs['logits'].detach().cpu().squeeze()
    
    
    stride = max_length - overlap
    chunks_ids = []
    chunks_mask = []
    
    for start in range(0, len(input_ids), stride):
        end = start + max_length
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        
        
        if len(chunk_ids) < max_length:
            pad_len = max_length - len(chunk_ids)
            chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
            chunk_mask = torch.cat([chunk_mask, torch.zeros(pad_len, dtype=torch.long)])
        
        chunks_ids.append(chunk_ids)
        chunks_mask.append(chunk_mask)
        
        if end >= len(input_ids):
            break
    
    
    batch_ids = torch.stack(chunks_ids).to(device)
    batch_mask = torch.stack(chunks_mask).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        preds = outputs['logits'].detach().cpu() 
    
    
    return preds.mean(dim=0).squeeze()


model_path = "./deberta-MERGE-tuned-prod"
dataset_path = "finetune/MERGE_PREPPED"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
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


print(f"{'Metric':<10} | {'MAE':<10} | {'Max Error':<10}")
print(f"{'Arousal':<10} | {sum(arousal_errors)/len(arousal_errors):<10.4f} | {max(arousal_errors):<10.4f}")
print(f"{'Valence':<10} | {sum(valence_errors)/len(valence_errors):<10.4f} | {max(valence_errors):<10.4f}")