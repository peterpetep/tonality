import torch
import random
from transformers import AutoTokenizer
from datasets import DatasetDict

from finetune.regression_model import RegressionModel
from safetensors.torch import load_file


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

#rand
dataset = DatasetDict.load_from_disk(dataset_path)
test_split = dataset['test']
random_idx = random.randint(0, len(test_split) - 1)
sample = test_split[random_idx]

lyrics = sample["Lyrics"]
true_arousal = sample["Arousal"]
true_valence = sample["Valence"]

print(f"Lyric: {lyrics[:100]}...")

# Tokenize inputs
inputs = tokenizer(
    lyrics, 
    truncation=True, 
    padding='max_length', 
    max_length=512, 
    return_tensors="pt"
).to(device)

with torch.no_grad():

    outputs = model(**inputs)
    print(outputs)
    preds = outputs['logits'].detach().cpu().squeeze()

   
    pred_arousal, pred_valence = preds[0].item(), preds[1].item()

print(f"{'Metric':<10} | {'Predicted':<10} | {'Actual':<10} | {'Error':<10}")
print(f"{'Arousal':<10} | {pred_arousal:<10.4f} | {true_arousal:<10.4f} | {abs(pred_arousal - true_arousal):<10.4f}")
print(f"{'Valence':<10} | {pred_valence:<10.4f} | {true_valence:<10.4f} | {abs(pred_valence - true_valence):<10.4f}")