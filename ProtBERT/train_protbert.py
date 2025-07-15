import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import DatasetDict, Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import wandb
import os

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
    RUN_ID = "tok_v1_run2"
    PROJECT_NAME = "ProtBERT_standalone"    # replace with your project name
    ENTITY = ""           # replace with your W&B entity
    OUTPUT_DIR = "./results_" + RUN_ID
    PROTBERT_CHECKPOINT = "" # replace with the protbert checkpoint to fine-tune

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
        dataset = DatasetDict.load_from_disk("tok_dataset_v1")
        # Create protbert binary classification model
        model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", num_labels=2)

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
        
        #Finetune the whole model

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=False,
            num_train_epochs=10,
            per_device_train_batch_size=16, 
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5, 
            weight_decay=0.01,
            save_total_limit=2,       # keep last 2 checkpoints
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
