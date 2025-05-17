#!/bin/bash

# ── Step 0: Install required packages ───────────────────────────────────────
pip install -q transformers datasets accelerate sentencepiece pandas gradio bert_score rouge_score

# ── Step 1: Set environment ────────────────────────────────────────────────
export WANDB_DISABLED=true

MODEL_NAME="google/flan-t5-base"
DATA_FILE="ami_gpt35_multitask.jsonl"
OUTPUT_DIR="flan_ami_multitask_final"
MODEL_PATH="$OUTPUT_DIR"

TEST_SPLIT=0.1
BATCH_SIZE=4
EPOCHS=3
LR=1e-5

# ── Step 2: Fine-tune FLAN-T5 ───────────────────────────────────────────────
python - <<END
import os, gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

model = AutoModelForSeq2SeqLM.from_pretrained("$MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained("$MODEL_NAME")

dataset = load_dataset("json", data_files="$DATA_FILE")['train']
split = dataset.train_test_split(test_size=$TEST_SPLIT, seed=42)
train_set = split['train']

def preprocess(batch):
    inputs = tokenizer(batch['input'], truncation=True, padding="max_length", max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch['output'], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_tok = train_set.map(preprocess, batched=True)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir="$OUTPUT_DIR",
    per_device_train_batch_size=$BATCH_SIZE,
    learning_rate=$LR,
    num_train_epochs=$EPOCHS,
    predict_with_generate=True,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    fp16=torch.cuda.is_available()
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    tokenizer=tokenizer,
    data_collator=collator,
)
trainer.train()

model.save_pretrained("$MODEL_PATH")
tokenizer.save_pretrained("$MODEL_PATH")
END

# ── Step 3: Launch pipeline UI ──────────────────────────────────────────────
python - <<END
import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr

MODEL_DIR = "$MODEL_PATH"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

def summarize_transcript(transcript):
    prompt = f"Summarize the following meeting transcript:\n{transcript}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=180)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_qa_pairs(transcript, summary):
    prompt = f"Generate 5 question-answer pairs from this meeting transcript:\n{transcript}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        qa_list = json.loads(decoded) if decoded.strip().startswith("[") else [{"question": "Summarize this meeting", "answer": summary}]
    except:
        qa_list = [{"question": "Summarize this meeting", "answer": summary}]
    return qa_list

def answer_question(transcript, summary, question):
    prompt = f"Answer the question based on the meeting transcript and summary.\n\nTranscript:\n{transcript}\n\nSummary:\n{summary}\n\nQuestion: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_pipeline(transcript, question):
    summary = summarize_transcript(transcript)
    if not question.strip():
        qa_pairs = generate_qa_pairs(transcript, summary)
        qa_str = "\n\n".join(f"Q: {q['question']}\nA: {q['answer']}" for q in qa_pairs)
        answer = ""
    else:
        qa_str = ""
        answer = answer_question(transcript, summary, question)
    return summary, qa_str, answer

gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(lines=8, label="Meeting Transcript"),
        gr.Textbox(lines=1, label="Your Question (optional)")
    ],
    outputs=[
        gr.Textbox(lines=6, label="\ud83d\udcdd Summary"),
        gr.Textbox(lines=8, label="\ud83d\udccb QA Pairs"),
        gr.Textbox(lines=3, label="\ud83d\udcac Answer")
    ],
    title="Meeting Summarizer + QA (FLAN-T5)",
    description="Paste a transcript to get a summary and QA pairs. Or ask a custom question about the meeting.",
    allow_flagging="never"
).launch()
END
