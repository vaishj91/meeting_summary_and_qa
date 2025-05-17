#!/bin/bash

# ── Step 0: Install Required Dependencies ─────────────────────────────────────
pip install -q openai pandas datasets tqdm

# ── Step 1: Generate Summaries and QA using GPT-3.5 ───────────────────────────
python - <<END
import openai, pandas as pd, json, re
from datasets import load_dataset
from tqdm import tqdm

# SET YOUR API KEY
openai.api_key = "your_openai_api_key_here"  # ← Replace before running

# Load the AMI corpus
df = load_dataset("nghiamkqn/AMI_corpus_extracted", split="train").to_pandas()
df = df[["meeting_id", "dialogue"]]
df["dialogue"] = df["dialogue"].astype(str)

# Clean text
df = df.dropna(subset=["dialogue"])
df = df[df["dialogue"].str.strip().astype(bool)].reset_index(drop=True)

# Prompt: Generate Summary
def generate_summary(text):
    prompt = f"""You are a helpful assistant that summarizes multi-party meeting transcripts.

Summarize the transcript below into a coherent narrative (200–250 words). Avoid using speaker names like A:, B:. Focus on key decisions, blockers, participants, and action items.

Transcript:
{text}

Summary:"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summary = res.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return ""

# Generate summaries with filter
summaries = []
for text in tqdm(df["dialogue"], desc="Generating Summaries"):
    s = generate_summary(text)
    if 150 <= len(s.split()) <= 300 and not re.search(r"[A-Z]:", s):  # Filter summary
        summaries.append(s)
    else:
        summaries.append("")  # mark for deletion
df["summary"] = summaries
df = df[df["summary"].str.strip().astype(bool)].reset_index(drop=True)

# Prompt: Generate QA Pairs
def generate_qa(text, summary):
    prompt = f"""You are a question generation assistant.

Based on the meeting transcript and its summary below, generate exactly 4 useful question–answer pairs.

Return ONLY a JSON list of 4 dictionaries, each with "question" and "answer" keys. Do NOT include any extra text.

Transcript:
{text}

Summary:
{summary}

Output:"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        output = res.choices[0].message.content.strip()
        parsed = json.loads(output)
        if isinstance(parsed, list) and len(parsed) == 4 and all("question" in qa and "answer" in qa for qa in parsed):
            return parsed
    except:
        pass
    return [{"question": "What was the main discussion?", "answer": summary}] * 4

# Generate QA pairs
qa_pairs = []
for i in tqdm(range(len(df)), desc="Generating QA Pairs"):
    qas = generate_qa(df.iloc[i]["dialogue"], df.iloc[i]["summary"])
    qa_pairs.append(qas)
df["qa_pairs"] = qa_pairs

# Save as pickle
df.to_pickle("ami_with_summaries_and_qa.pkl")
print("Saved dataset: ami_with_summaries_and_qa.pkl")

# ── Step 2: Convert to jsonl for fine-tuning ─────────────────────────────────
records = []
for _, row in df.iterrows():
    records.append({
        "input": f"Summarize the following meeting transcript:\n{row['dialogue']}",
        "output": row["summary"]
    })
    for qa in row["qa_pairs"]:
        records.append({
            "input": f"Q: {qa['question']}\n\nTranscript:\n{row['dialogue']}",
            "output": qa["answer"]
        })

with open("ami_gpt35_multitask.jsonl", "w") as f:
    for r in records:
        json.dump(r, f)
        f.write("\n")

print("Saved training file: ami_gpt35_multitask.jsonl")
END
