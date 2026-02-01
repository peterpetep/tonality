from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModel
import torch.nn as nn
import torch

class RegressionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.deberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        ans = out.last_hidden_state[:, 0] #cls
        logits = self.regressor(ans)
        

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def data_collator(features):
    #print(features)
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'labels': torch.stack([
            torch.tensor([f['Arousal'], f['Valence']], dtype=torch.float32) 
            for f in features
        ])
    }
    return batch


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    arousal_mse = mean_squared_error(labels[:, 0], predictions[:, 0])
    valence_mse = mean_squared_error(labels[:, 1], predictions[:, 1])
    arousal_mae = mean_absolute_error(labels[:, 0], predictions[:, 0])
    valence_mae = mean_absolute_error(labels[:, 1], predictions[:, 1])
    
    overall_mse = mean_squared_error(labels, predictions)
    overall_mae = mean_absolute_error(labels, predictions)
    
    return {
        'mse': overall_mse,
        'mae': overall_mae,
        'arousal_mse': arousal_mse,
        'arousal_mae': arousal_mae,
        'valence_mse': valence_mse,
        'valence_mae': valence_mae,
    }