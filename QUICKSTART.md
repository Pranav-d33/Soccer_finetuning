# ⚡ Quick Start Guide

## 🎯 What You Have

✅ **3.1 Million instruction-based training samples** for fine-tuning soccer tactical reasoning models

- `football_train.jsonl` - 2.8M training samples
- `football_val.jsonl` - 312k validation samples
- 5 balanced instruction types (20% each)
- 285 unique teams across 11 European leagues
- 25,629 unique matches from 2008-2016

---

## 🚀 Get Started in 3 Steps

### Step 1: Verify the Dataset
```bash
python validate_dataset.py
```

This shows you:
- Dataset statistics
- Quality metrics
- Distribution across instruction types
- Sample examples

### Step 2: Load in Your Framework

**HuggingFace Transformers:**
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "football_train.jsonl",
    "validation": "football_val.jsonl"
})

# Access samples
print(dataset["train"][0])
```

**Quick peek at data:**
```bash
# Show first sample (Linux/Mac)
head -1 football_train.jsonl | python -m json.tool

# Or with PowerShell (Windows)
Get-Content football_train.jsonl -Head 1 | ConvertFrom-Json
```

### Step 3: Start Fine-Tuning

**Example with Unsloth & LoRA:**
```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, TrainingArguments

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/phi-2-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth"
)

# Load dataset
dataset = load_dataset("json", data_files="football_train.jsonl")

# Train
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "instruction",
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,
        warmup_steps = 100,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 100,
        output_dir = "soccer_model",
        evaluation_strategy = "steps",
        eval_steps = 1000,
    ),
)

trainer.train()
```

---

## 📋 Dataset Format

Each line is a complete JSON object:

```json
{
  "instruction": "Explain why Barcelona used a balanced, controlled pace approach in this match.",
  "input": "Match: Barcelona vs Real Madrid | Season: 2012/2013 | League: La Liga | Score: 2-1",
  "output": "Barcelona employed a balanced, controlled pace strategy to dominate possession and control the game tempo. Their short-pass oriented, defensive discipline approach was designed to neutralize Real Madrid's strengths and build attacks methodically. With the final score of 2-1, their tactical execution proved effective in key moments.",
  "metadata": {
    "match_id": 12345,
    "home_team": "Barcelona",
    "away_team": "Real Madrid",
    "type": "tactical_reasoning"
  }
}
```

---

## 5️⃣ Instruction Types

### 1. Tactical Reasoning (20%)
Why teams use specific formations and tactics

### 2. Match Summary (20%)
High-level narrative and outcome description

### 3. Formation Comparison (20%)
Tactical contrasts between teams

### 4. Predictive Analysis (20%)
What-if scenarios and strategic alternatives

### 5. Player Analysis (20%)
Team and individual performance evaluation

---

## 📊 By The Numbers

| Metric | Value |
|--------|-------|
| Total Samples | 3,126,500 |
| Training | 2,813,850 |
| Validation | 312,650 |
| Avg Output | 398 characters |
| Unique Teams | 285 |
| Unique Matches | 25,629 |
| Leagues | 11 European |
| Years | 2008-2016 |

---

## 🛠️ Utility Scripts

**Database exploration:**
```bash
python explore_data.py
```

**Dataset validation:**
```bash
python validate_dataset.py
```

**Regenerate dataset** (if needed):
```bash
python generate_dataset.py
```

---

## 💡 Pro Tips

1. **Start small** - Test with a subset first
2. **Monitor validation loss** - Use early stopping
3. **Adjust batch size** based on your GPU RAM
4. **Use gradient accumulation** for larger effective batch sizes
5. **LoRA is efficient** - Try it with base models
6. **Data is balanced** - All instruction types equally represented

---

## 📚 Documentation

- **DATASET_SUMMARY.md** - Full overview & results
- **PREPROCESSING_GUIDE.md** - Detailed technical documentation
- **EXECUTION_GUIDE.md** - Step-by-step instructions

---

## ⚙️ Recommended Settings

**For Phi-2 (2.7B):**
- Batch: 16
- Learning Rate: 2e-4
- Epochs: 3
- LoRA Rank: 16

**For Mistral (7B):**
- Batch: 8
- Learning Rate: 1e-4
- Epochs: 2
- LoRA Rank: 32

**For Larger Models (13B+):**
- Batch: 4
- Learning Rate: 5e-5
- Epochs: 1-2
- LoRA Rank: 64

---

## 🎓 What Model Learns

✅ Tactical reasoning and formations  
✅ Match analysis and summarization  
✅ Strategy comparison  
✅ Performance evaluation  
✅ Soccer domain knowledge  

---

## ❓ FAQ

**Q: Can I use just the training set?**
A: Yes, but validation set is recommended for monitoring.

**Q: Can I combine samples?**
A: Yes, the data is independent. Feel free to mix/filter.

**Q: How do I update the dataset?**
A: Run `generate_dataset.py` again with different parameters.

**Q: Is this production-ready?**
A: Yes! All quality checks passed, no missing values.

**Q: Can I use this with commercial models?**
A: Check terms of service for the model + data usage rights.

---

🚀 **You're all set! Start fine-tuning!**
