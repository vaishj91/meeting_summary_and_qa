# 🧠 Meeting Summarizer + QA System

This project implements a multi-task NLP system that performs abstractive summarization and question answering (QA) on meeting transcripts using fine-tuned FLAN-T5 models. A Gradio interface allows users to interactively input a transcript, generate a summary, view QA pairs, or ask a follow-up question.

---

## Features

- Generates 200–250 word summaries from multi-party meeting transcripts
- Produces 5 question–answer pairs per meeting
- Supports custom question input and real-time answers
- Fine-tunes FLAN-T5 on both summarization and QA tasks
- Gradio-based web interface for easy use

---
## ⚙️ Requirements

- Python 3.8+
- GPU support recommended (for training)
- Hugging Face Transformers
- Gradio

---

## 📊 Evaluation (Summarization)

| Metric           | Score    |
|------------------|----------|
| ROUGE-1          | 0.3457   |
| ROUGE-2          | 0.1117   |
| ROUGE-L          | 0.2488   |
| BERTScore F1     | 0.8383   |
| Cosine Relevance | 0.5492   |

📌 **QA EM / F1**: 0.0 — early proof of concept; performance expected to improve with larger and cleaner QA dataset.

---

## 🛠️ Technologies Used

- Python 3.10
- Hugging Face Transformers
- Gradio
- OpenAI GPT-3.5 (for dataset generation)
- ROUGE, BERTScore (evaluation)
- FLAN-T5 (`google/flan-t5-base`)

---

## 📦 Dataset

- **Source**: AMI Meeting Corpus
- **Synthetic Generation** via GPT-3.5:
  - ~100 summaries (200–250 words)
  - ~500 QA pairs
- Converted QA format: JSON → SQuAD-style
- Combined for multi-task fine-tuning of FLAN-T5

---

## ✨ Future Improvements

- Improve QA performance via multi-task joint tuning
- Add support for live transcripts (Zoom, Teams)
- Introduce human evaluation metrics for summaries
- Domain adaptation for enterprise language
- Add NER for speaker-aware or personalized outputs

---

## ▶️ How to Run

### 🔧 Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for training)
- Bash terminal (for running shell script blocks)

---

### 🧩 Step-by-Step

1. **Clone the repo**  
   ```bash
   git clone https://github.com/vaishj91/meeting_summary_and_qa/tree/main
   cd meeting-summarizer-qa

2. **Place your dataset**
Ensure your input file (e.g., ami_gpt35_multitask.jsonl) is in the project root. This file should contain a JSONL format with "input" and "output" keys.

3. **Run the full fine-tuning + deployment pipeline**
Save the script (you shared) as run_pipeline.sh, then run:

```
bash run_pipeline.sh
```

4. **Use the Web UI**
After training, a Gradio app will launch at:
http://localhost:7860
Paste a transcript, optionally type a question, and hit submit.

---

## 📁 File Structure

```bash
├── 01_dataset_generation.ipynb      # Generates synthetic summaries and QA pairs using GPT
├── 03_training&eval.ipynb           # Trains and evaluates
├── 02_fine-tuning&pipeline.ipynb    # Fine-tunes model, loads it, and launches Gradio interface [entrypoint]
├── ami_with_summaries_and_qa.pkl    # Serialized multitask dataset (summaries + QA)
└── README.md
