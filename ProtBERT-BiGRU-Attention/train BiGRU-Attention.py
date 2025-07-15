import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import DatasetDict, Dataset
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
import re

import wandb
import os

# Stable multihead attention
class MultiHeadAttention(nn.Module):
  def __init__(self, hidden_size, head_size, num_heads):
    super().__init__()
    self.hidden_size = hidden_size
    self.head_size = head_size
    self.num_heads = num_heads

    self.w_u = nn.Linear(hidden_size, head_size * num_heads)

  def forward(self, x):
    B, L, H = x.size()
    u = self.w_u(x)
    u = u.view(B, L, self.num_heads, self.head_size) # [B, L, n, Dh]

    A = torch.einsum("blnd,bknd->blnk", u, u) # Attention matrices [B, L, n, L]
    A = A.contiguous() # important since einsum leaves A discontigous so view cannot be used

    # Substract the max value for each batch example to prevent overflows in the exp (stabilization).
    A = A.view(B, L, self.num_heads*L) - A.view(B, -1).max(dim=1, keepdim=True).values.unsqueeze(-1)
    exp_A = torch.exp(A)
    
    alpha = torch.einsum("bln->bl",exp_A) / torch.einsum("bln->b", exp_A).unsqueeze(-1)

    attended = torch.einsum("bl,blh->bh", alpha, x)
    return attended

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, 
                     num_layers=num_layers, batch_first=True, 
                     dropout=dropout, bidirectional=True)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = F.relu(self.linear(x))
        return x

class ProtBERT_BiGRU_Attention(nn.Module):
    def __init__(self, protbert_model, input_size, hidden_size, attnhead_size, num_attnheads, dropout=0.0, gru_layers=1):
        super().__init__()
        self.protbert = protbert_model
        self.bigru = BiGRU(input_size, hidden_size, num_layers=gru_layers)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(hidden_size, attnhead_size, num_attnheads)
        self.classifier = nn.Linear(hidden_size, 2)

        # Initialize weights with He initialization to prevent exploding gradients
        self.init_weights()

    def init_weights(self):
        def init_weights_(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.kaiming_uniform_(param, nonlinearity='sigmoid')
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
        
        self.bigru.apply(init_weights_)
        self.attention.apply(init_weights_)
        self.classifier.apply(init_weights_)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        if input_ids.size(1) != 1003:
           print("Wrong number of input ids:", len(input_ids))
           return
        
        # Use precomputed embeddings directly
        outputs = self.protbert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        mask = torch.ones(1003, dtype=torch.bool)
        mask[[0, 501, 1002]] = False
        embeddings = outputs.last_hidden_state[:, mask, :]

        x = self.bigru(embeddings)  # Shape: (batch_size, seq_len, hidden_size)
        x = self.dropout(x)
        x = self.attention(x)  # Ensure attention handles sequence dimension
        logits = self.classifier(x)  # Shape: (batch_size, 1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            # Ensure logits and labels are 1D for BCEWithLogitsLoss
            #loss = loss_fct(logits.squeeze(), labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

# Accuracy as evaluation metric
import evaluate
from sklearn.metrics import average_precision_score, roc_auc_score
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Check if any logit is NaN
    if np.isnan(logits).any():
        print("Logits are NaN")
    
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)["accuracy"]
    # calculate Precision-recall AUC and ROC AUC:
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    pos_probs = probs[:, 1] # probability of the positive class

    pr_auc = average_precision_score(labels, pos_probs)
    roc_auc = roc_auc_score(labels, pos_probs)
    
    return {
        "accuracy": acc,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }

def main():
    RUN_ID = "stable_full1"
    PROJECT_NAME = "ProtBERT-PPI"    # replace with your project name
    ENTITY = ""           # replace with your W&B entity
    OUTPUT_DIR = "./results_" + RUN_ID
    PROTBERT_CHECKPOINT = "ProtBERT_FT-PPI/checkpoint-2295"

    #os.environ["WANDB_LOG_MODEL"] = "checkpoint" #to log the model to wandb as an artifact
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    wandb.login()

    
    with wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        id=RUN_ID,
        resume="allow"
    ) as run:
        # Set precision to high (instead of highest) as this improves speed
        torch.set_float32_matmul_precision('high')

        # Load Dataset
        #dataset = DatasetDict.load_from_disk("tokenized_dataset")
        dataset = DatasetDict.load_from_disk("tokenized_dataset_full") #stable_full
        # Create aggregated model
        protbert_model = BertModel.from_pretrained(PROTBERT_CHECKPOINT)
        model = ProtBERT_BiGRU_Attention(protbert_model, 1024, 41, 32, 2, dropout=0.2403)

        # Check if there is any checkpoint
        checkpoint_dir = None
        if not os.path.isdir(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
            try:
                art = run.use_artifact(f"{ENTITY}/{PROJECT_NAME}/checkpoint-{RUN_ID}:latest", type="model")
                print(f"Downloading artifact model-{RUN_ID}:latest...")
                checkpoint_dir = art.download()
            except Exception as e:
                print("No existing artifact found, will train from scratch.", e)
        else:
            checkpoint_dir = True
        
        # Do not train the protbert model
        model.protbert.requires_grad_(False)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=False,
            num_train_epochs=50,
            per_device_train_batch_size=32, #Arbitrary, as the paper does not say
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=0.0017, # According to the paper, but using AdamW
            weight_decay=0.0124,
            save_total_limit=4,       # keep last 2 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_strategy="steps",
            logging_steps=10,
            report_to="wandb",
            torch_compile=True, # Compile the model. It will go much faster. I will use the default mode for torch compile
            #torch_compile_mode="reduce-overhead" 
            # This option tries to generate cudagraphs (which are a lot of speedup on top of torch compile). However i get this error:
            # skipping cudagraphs due to skipping cudagraphs due to cpu device (index_put)
            # Probably some problem with the model or the cpu that prevents it from being able to compile it into cuda.
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
        )

        trainer.train(resume_from_checkpoint=checkpoint_dir) # If not found in wandb resume from local (True)

main()
