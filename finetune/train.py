from pathlib import Path
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

from finetune.regression_model import RegressionModel, compute_metrics, data_collator
from safetensors.torch import load_file

existing_model = "./deberta-MERGE-tuned-prod"
emod_path = Path(existing_model)

model_name = "microsoft/deberta-v3-base"

if emod_path.exists():
    tokenizer = AutoTokenizer.from_pretrained(existing_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = DatasetDict.load_from_disk("finetune/MERGE_PREPPED")

def tokenize(examples):
    tokenized = tokenizer(examples["Lyrics"], truncation=True, padding='max_length', max_length=512)
    tokenized["Arousal"] = examples["Arousal"]
    tokenized["Valence"] = examples["Valence"]
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Arousal', 'Valence'])

model = RegressionModel(model_name)

if emod_path.exists():
    weights = load_file(f"{existing_model}/model.safetensors")
    model.load_state_dict(weights)

model.to(torch.float32)
model.to("mps")

training_args = TrainingArguments(
    output_dir="./deberta-MERGE-tuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,
    warmup_steps=500,
    report_to="tensorboard",
    fp16=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.save_model("./deberta-MERGE-tuned-prod")
tokenizer.save_pretrained(".//deberta-MERGE-tuned-prod")

print("done")