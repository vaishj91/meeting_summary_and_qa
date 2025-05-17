#!/bin/bash

# Install required packages
pip install -q transformers datasets accelerate sentencepiece pandas bert_score rouge_score

# Disable Weights & Biases logging
export WANDB_DISABLED=true

# Define model and dataset paths
MODEL_NAME="google/flan-t5-base"
DATA_FILE="ami_gpt35_multitask.jsonl"
OUTPUT_DIR="flan_ami_multitask"
MODEL_PATH="$OUTPUT_DIR"

# Train-test split seed and config
TEST_SPLIT=0.1
BATCH_SIZE=4
EPOCHS=3
LR=1e-5

# Python training script
python - <<END
import os, gc
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Load base model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("$MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained("$MODEL_NAME")

# Load dataset
dataset = load_dataset("json", data_files="$DATA_FILE")['train']
split_data = dataset.train_test_split(test_size=$TEST_SPLIT, seed=42)
train_data, test_data = split_data['train'], split_data['test']

# Preprocess
max_input_len, max_output_len = 512, 128
def preprocess(batch):
    inputs = tokenizer(batch['input'], truncation=True, padding="max_length", max_length=max_input_len)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch['output'], truncation=True, padding="max_length", max_length=max_output_len)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_train = train_data.map(preprocess, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="$OUTPUT_DIR",
    per_device_train_batch_size=$BATCH_SIZE,
    learning_rate=$LR,
    num_train_epochs=$EPOCHS,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()

# Save model
model.save_pretrained("$MODEL_PATH")
tokenizer.save_pretrained("$MODEL_PATH")
END

# Python evaluation script
python - <<END
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from tqdm import tqdm

model = AutoModelForSeq2SeqLM.from_pretrained("$MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained("$MODEL_PATH")
dataset = load_dataset("json", data_files="$DATA_FILE")['train']
test_set = dataset.train_test_split(test_size=$TEST_SPLIT, seed=42)['test']

results = []
for example in test_set:
    inputs = tokenizer(example['input'], return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({"input": example["input"], "gold_output": example["output"], "generated_output": pred})

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("qa_test_split_outputs.csv", index=False)

# BERTScore
P, R, F1 = bertscore(df["generated_output"].tolist(), df["gold_output"].tolist(), lang="en")
df["BERTScore_F1"] = F1

# ROUGE
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
r1, r2, rl = [], [], []
for ref, pred in tqdm(zip(df["gold_output"], df["generated_output"]), total=len(df)):
    scores = rouge.score(ref, pred)
    r1.append(scores["rouge1"].fmeasure)
    r2.append(scores["rouge2"].fmeasure)
    rl.append(scores["rougeL"].fmeasure)
df["ROUGE-1"], df["ROUGE-2"], df["ROUGE-L"] = r1, r2, rl

df.to_csv("qa_test_split_outputs_with_metrics.csv", index=False)
print("Evaluation complete. Results saved to qa_test_split_outputs_with_metrics.csv")
END