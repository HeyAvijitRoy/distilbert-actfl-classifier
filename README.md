# DistilBERT ACTFL-style English Writing Proficiency Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-black?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange?logo=huggingface&logoColor=white)](https://huggingface.co/transformers/) [![Gradio](https://img.shields.io/badge/Gradio-UI-green?logo=gradio&logoColor=white)](https://gradio.app/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**A self-hosted pipeline that trains a DistilBERT-based classifier to map English learner essays to ACTFL-like proficiency buckets (10 classes: Novice Low → Superior).** Includes data preparation, augmentation from Hugging Face CommonLit readability data, training with class-weighted loss, evaluation with real confusion matrices, and a simple Gradio demo for inference.

---

## 📋 Table of Contents

- [Quick Start (PowerShell)](#quick-start-powershell)
- [At-a-Glance](#at-a-glance)
- [Repository Structure](#repository-structure)
- [Data Sources & Placement](#data-sources--placement)
- [ACTFL Proficiency Levels](#actfl-proficiency-levels)
- [Model Card & Performance](#model-card--performance)
- [Architecture & Training](#architecture--training)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [1. Prepare Labeled Data](#1-prepare-labeled-data)
  - [2. (Optional) Augment with CommonLit](#2-optional-augment-with-commonlit)
  - [3. Train](#3-train)
  - [4. Evaluate](#4-evaluate)
  - [5. Interactive Inference](#5-interactive-inference)
- [Technical Implementation](#technical-implementation)
- [⚠️ Limitations & Reliability](#️-limitations--reliability)
- [Contributing & Data Collection](#contributing--data-collection)
- [Troubleshooting](#troubleshooting)
- [License & Contact](#license--contact)

---

## 🚀 Quick Start (PowerShell)

**1. Create & activate virtual environment:**
```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   # one-time if needed
.\venv\Scripts\Activate.ps1
```

**2. Install dependencies:**
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**3. Prepare labeled data:**
```powershell
python script.py
# → creates: asap_actfl_labeled.csv
```

**4. (Optional) Augment with CommonLit high-proficiency samples:**
```powershell
python prepare_hf_augmented.py
# → creates: asap_plus_hf.csv
```

**5. Train model:**
```powershell
python train.py
# → generates: model_output/ (checkpoints) and distilbert-actfl-english/ (final model)
```

**6. Evaluate:**
```powershell
python eval.py
# → prints confusion matrix, metrics, per-class accuracy
```

**7. Launch interactive demo:**
```powershell
python app.py
# → opens Gradio UI at http://localhost:7860
```

---

## 📌 At-a-Glance

What's in this repository:

✅ **Data pipeline**: ASAP raw data → ACTFL-labeled CSV with 10 proficiency bins  
✅ **Augmentation**: Optional CommonLit Ease-of-Readability integration for high-proficiency classes  
✅ **Training**: Hugging Face `Trainer` with class-weighted loss to handle imbalance  
✅ **Evaluation**: Confusion matrix, per-class accuracy, macro F1-score  
✅ **Inference UI**: Gradio app for single-essay predictions  
✅ **Model artifacts**: Saved model (`distilbert-actfl-english/`) and checkpoints (`model_output/`)  
✅ **Reproducibility**: All hyperparameters and random seeds explicit in `train.py`

---

## 📁 Repository Structure

```
.
├─ asap-aes/                           # Place ASAP TSV files here (training_set_rel3.tsv, etc.)
├─ asap_actfl_labeled.csv              # Created by script.py
├─ asap_plus_hf.csv                    # Created by prepare_hf_augmented.py (optional)
├─ script.py                           # Label ASAP rubric scores → ACTFL bins
├─ prepare_hf_augmented.py             # Augment with CommonLit readability samples
├─ train.py                            # Training script (HF Trainer, class-weighted loss)
├─ eval.py                             # Evaluation script (confusion matrix + metrics)
├─ app.py                              # Gradio inference UI
├─ distilbert-actfl-english/           # Final saved model (after training)
├─ model_output/                       # Trainer checkpoints and logs
├─ requirements.txt                    # Python dependencies
├─ LICENSE
├─ MODEL_CARD.md                       # Detailed model card (datasets, metrics)
└─ README.md                           # This file
```

---

## 📊 Data Sources & Placement

| Dataset | Source | Purpose | Size |
|---------|--------|---------|------|
| **ASAP AES** | [Kaggle Competition](https://www.kaggle.com/competitions/asap-aes) | Primary labeled essays (rubric scores) | ~12,976 examples |
| **CommonLit Ease-of-Readability** | [Hugging Face](https://huggingface.co/datasets/casey-martin/CommonLit-Ease-of-Readability) | Augment high-proficiency classes (Flesch Reading Ease filter) | ~355 Superior + ~354 Advanced High |

**Setup:**
- Download ASAP training/validation TSV files and place in `asap-aes/` folder
- CommonLit is auto-downloaded by `prepare_hf_augmented.py` if augmentation is needed

---

## 🗣️ ACTFL Proficiency Levels

The ACTFL Proficiency Guidelines framework used in this project (10 levels):

| Level | Tier | Description |
|-------|------|-------------|
| **Novice Low** | Beginner | Limited vocabulary, frequent errors, simple sentences |
| **Novice Mid** | Beginner | Emerging patterns, isolated words/phrases, basic structures |
| **Novice High** | Beginner | Simple sentences, common topics, comprehensible with effort |
| **Intermediate Low** | Intermediate | Expanded vocabulary, short paragraphs, generally understandable |
| **Intermediate Mid** | Intermediate | Consistent structures, varied topics, occasional errors |
| **Intermediate High** | Intermediate | Complex ideas, detailed descriptions, minor errors |
| **Advanced Low** | Advanced | Abstract concepts, varied register, sophisticated vocabulary |
| **Advanced Mid** | Advanced | Nuanced expression, cultural references, accurate syntax |
| **Advanced High** | Advanced | Superior organization, idioms, near-native proficiency |
| **Superior** | Mastery | Native-like proficiency, complex discourse, exceptional clarity |

---

## 📋 Model Card & Performance

### Best Reported Evaluation (Final Run)

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.9291 (92.91%) |
| **Macro F1** | 0.4417 |
| **Eval Loss** | 0.2392 |

### Per-Class Accuracy (Real Results)

| Label | Correct / Total | Accuracy |
|-------|--------|----------|
| Novice Low | 1819 / 1833 | **99.24%** |
| Novice Mid | 346 / 376 | **92.02%** |
| Novice High | 112 / 142 | **78.87%** |
| Intermediate Low | 65 / 97 | **67.01%** |
| **Intermediate Mid** | **0 / 15** | **0.00%** ⚠️ |
| Intermediate High | 27 / 55 | **49.09%** |
| Advanced Low | 43 / 61 | **70.49%** |
| **Advanced Mid** | **0 / 13** | **0.00%** ⚠️ |
| **Advanced High** | **0 / 3** | **0.00%** ⚠️ |
| **Superior** | **0 / 1** | **0.00%** ⚠️ |

**⚠️ Key Observation:** High overall accuracy (92.91%) is **heavily driven by Novice Low dominance** (1833/2596 = 70% of evaluation set). Higher-proficiency classes are severely under-represented and have poor per-class accuracy. See [Limitations & Reliability](#️-limitations--reliability) for critical details.

### Confusion Matrix (Real Data)

```
              Predicted →
              NL  NM  NH  IL  IM  IH  AL  AM  AH  Su
Novice Low    1819 14   0   0   0   0   0   0   0   0
Novice Mid      4 346  25   1   0   0   0   0   0   0
Novice High     2  10 112  18   0   0   0   0   0   0
Intermediate L  3   0  27  65   1   1   0   0   0   0
Intermediate M  0   0   1   0   0  11   3   0   0   0
Intermediate H  0   0   0   2   0  27  26   0   0   0
Advanced Low    0   0   0   1   0  17  43   0   0   0
Advanced Mid    0   0   0   0   0   1  12   0   0   0
Advanced High   0   0   0   0   0   0   3   0   0   0
Superior        0   0   0   1   0   0   0   0   0   0
```

---

## 🏗️ Architecture & Training

### Model Specification

- **Base:** `distilbert-base-uncased` (66M parameters, 60% smaller than BERT)
- **Tokenizer:** DistilBERT WordPiece tokenizer
- **Output Head:** Linear layer (768 → 10 ACTFL classes)
- **Why DistilBERT?** Efficient, effective (97% BERT performance), production-ready, pre-trained on 12GB+ English text

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Max Sequence Length | 256 tokens (~1000 words) |
| Batch Size | 8 (train & eval) |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Loss Function | Class-weighted CrossEntropy |
| Optimization | AdamW (Hugging Face Trainer) |
| Train / Validation Split | 80 / 20 |

### Class Weighting

To handle imbalance, the model uses inverse-frequency weighting:

$$\text{weight}_c = \frac{\text{total samples}}{n\_\text{classes} \times \text{samples in class}_c}$$

This forces the model to allocate more learning capacity to underrepresented classes.

---

## 💾 Installation

### Prerequisites

- **Python:** 3.8 or higher
- **CUDA (Optional):** For GPU acceleration (NVIDIA GPU with CUDA 11.8+)
- **Git:** For cloning

### Setup Steps

```powershell
# Step 1: Clone
git clone <repository-url>
cd English

# Step 2: Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Step 3: Install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Step 4: Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}')"
```

**Key Dependencies:**
- `torch` — PyTorch framework
- `transformers` — Hugging Face models & Trainer
- `datasets` — HF datasets library
- `scikit-learn` — Evaluation metrics
- `pandas` — Data manipulation
- `gradio` — Web UI
- `numpy`, `matplotlib` — Numerics & plotting

---

## 📖 Usage Guide

### 1. Prepare Labeled Data

```powershell
python script.py
```

**What it does:**
- Reads ASAP TSV files from `asap-aes/`
- Bins `domain1_score` into 10 quantile buckets using `pd.qcut`
- Maps buckets to ACTFL labels: Novice Low, ..., Superior
- Outputs: `asap_actfl_labeled.csv` (essay + actfl_level columns)

**Output example:**
```csv
essay,actfl_level
"The sun is bright and I like it.",Novice Low
"Learning languages is important for communication.",Intermediate Mid
```

---

### 2. (Optional) Augment with CommonLit

```powershell
python prepare_hf_augmented.py
```

**What it does:**
- Downloads CommonLit readability dataset from Hugging Face
- Selects top 10% easiest passages → "Superior"
- Selects top 10-20% easiest passages → "Advanced High"
- Combines with ASAP data
- Outputs: `asap_plus_hf.csv` (~13,685 total examples)

**Use in training:**
Edit `train.py` line `data_files = {"data": "asap_actfl_labeled.csv"}` to point to `asap_plus_hf.csv`

---

### 3. Train

```powershell
python train.py
```

**What happens:**
1. Loads labeled CSV (80% train, 20% validation)
2. Tokenizes essays (max 256 tokens)
3. Initializes DistilBERT + 10-class head
4. Fine-tunes for 3 epochs with class-weighted loss
5. Saves checkpoints every 500 steps → `model_output/checkpoint-*`
6. Saves final model → `distilbert-actfl-english/`

**Estimated time:**
- GPU (RTX 3080): 2-3 hours
- CPU: 15-20 hours

---

### 4. Evaluate

```powershell
python eval.py
```

**Output:**
- Confusion matrix (rows = true labels, cols = predictions)
- Per-class accuracy table
- Overall accuracy, macro F1, loss
- Example: See [Model Card & Performance](#model-card--performance) above

---

### 5. Interactive Inference

```powershell
python app.py
```

**Then:**
1. Open browser to printed URL (usually `http://localhost:7860`)
2. Paste or type essay text
3. Get predicted ACTFL level + confidence scores

**Example input:**
```
I went to the store yesterday. I bought milk and bread.
The store was very big. I like shopping there.
```

**Example output:**
```
Predicted ACTFL Level: Novice Mid
Probabilities:
  Novice Low: 0.05
  Novice Mid: 0.82  ← highest
  Novice High: 0.10
  ...
```

---

## 🔬 Technical Implementation

### Tokenization Pipeline

```python
inputs = tokenizer(
    essay_text,
    truncation=True,          # Truncate to max_length
    padding="max_length",     # Pad to max_length
    max_length=256,           # 256 tokens ≈ ~1000 words
    return_tensors="pt"       # PyTorch tensors
)
# Returns: input_ids, token_type_ids, attention_mask
```

**Max length rationale:**
- 256 tokens covers ~99% of student essays
- Longer sequences increase memory & training time
- Shorter sequences lose information

### Inference (Forward Pass)

```python
# 1. Tokenize
inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")

# 2. Forward pass (no gradients)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # [batch, 10]

# 3. Probabilities
probs = torch.softmax(logits, dim=-1)  # sum = 1.0

# 4. Top prediction
pred_idx = torch.argmax(probs, dim=-1)
confidence = probs[0, pred_idx]

# 5. Map to ACTFL label
label = ACTFL_LABELS[pred_idx]
```

### Label Mapping (Index ↔ Label)

```python
ACTFL_LABELS = [
    "Novice Low",         # 0
    "Novice Mid",         # 1
    "Novice High",        # 2
    "Intermediate Low",   # 3
    "Intermediate Mid",   # 4
    "Intermediate High",  # 5
    "Advanced Low",       # 6
    "Advanced Mid",       # 7
    "Advanced High",      # 8
    "Superior"            # 9
]
```

---

## ⚠️ Limitations & Reliability

### 🚨 Critical Issue: Severe Class Imbalance

**The model's predictions are heavily skewed by training data distribution.** This is the single most important limitation to understand:

#### The Problem

- **Novice Low:** 1,833 examples (70% of eval set)
- **Novice Mid:** 376 examples (14%)
- **Intermediate High:** 55 examples (2%)
- **Superior:** 1 example (0.04%)

Result: **Novice Low dominates predictions.** Even though overall accuracy is 92.91%, this is almost entirely because the model learned to predict Novice Low accurately. Higher proficiency classes (Superior, Advanced High, Intermediate Mid) have **0% per-class accuracy** due to extreme under-representation.

#### Impact on Usage

✅ **Reliable (Safe to Use):**
- Classifying Novice-level writing (Novice Low/Mid)
- Filtering clearly beginner essays
- Research & development

❌ **Unreliable (NOT Safe):**
- Classifying Advanced or Superior writing
- High-stakes decisions (grading, placement)
- Production systems without human review
- Any classification without checking per-class accuracy in confusion matrix

### Class Imbalance: Why Weighting Isn't Enough

This implementation uses **class-weighted loss** during training:

```python
class_weights = torch.tensor([w1, w2, ..., w10])
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

**What it helps with:**
✅ Prevents collapse of minority classes entirely  
✅ Forces allocation of learning capacity to rare classes  
✅ Improves per-class F1 compared to unweighted baseline

**What it can't fix:**
❌ Cannot create information that doesn't exist  
❌ Cannot overcome 1800:1 sample ratio (Novice Low : Superior)  
❌ Limited by magnitude of imbalance  

**Reality:** With only 1 Superior example vs. 1,833 Novice Low, no amount of weighting can solve the fundamental data shortage.

### Why High Overall Accuracy is Misleading

```
Overall Accuracy = 92.91%
This looks great until you check per-class:

  Novice Low:      99.24% ✅ (1,833 examples)
  Novice Mid:      92.02% ✅ (376 examples)
  Novice High:     78.87% ⚠️ (142 examples)
  Intermediate Low: 67.01% ⚠️ (97 examples)
  Advanced Low:    70.49% ⚠️ (61 examples)
  
  Advanced High:    0.00% ❌ (only 3 examples)
  Superior:         0.00% ❌ (only 1 example)
  
The 92.91% average is weighted by volume, not represented equally across classes.
Macro F1 (0.44) is more honest — it shows average performance across all classes.
```

### Critical Recommendations

**For Research / Development:**
```
✅ Current model is suitable for:
├─ Exploratory data analysis
├─ Proof-of-concept prototypes
├─ Screening coarse Novice-level writing
└─ Understanding ACTFL classification task
```

**For Production / High-Stakes Use:**
```
❌ Current model is NOT suitable without:
├─ Rebalanced dataset (1,000+ examples per level minimum)
├─ Expert validation of edge cases
├─ Per-class accuracy guarantees
├─ Human review of Advanced/Superior classifications
└─ Documented fallback (escalate to human if confidence < X%)
```

### Path to Production

Before using this model in production, you MUST:

1. **Collect Balanced Dataset**
   - Goal: ≥1,000 essays per ACTFL level (10,000+ total)
   - Use stratified sampling or multi-source collection
   - Current: 2,596 total (highly imbalanced)

2. **Retrain on Balanced Data**
   - Run `train.py` with new dataset
   - Expected improvement: +5-10% per-class accuracy

3. **Validate on Independent Test Set**
   - Use k-fold cross-validation
   - Ensure high per-class accuracy (≥80% per level)

4. **Expert Review**
   - Have ACTFL professionals review edge cases
   - Calibrate confidence thresholds

5. **Document & Release v1.0**
   - Update this README with new metrics
   - Tag release, document limitations clearly
   - Publish reproduction steps

### Model Development Status

```
Current Phase: Early Exploration
├─ Dataset: 2,596 examples (imbalanced)
├─ Use cases: Research, POC, development
├─ Production ready: NO
└─ Recommended: For development use only with caveats

Next Phase: Balanced dataset v1 (planned)
├─ Target: 5,000 examples (500 per level)
├─ Augmentation: Text paraphrasing for rare classes
├─ Retraining: Full pipeline
└─ Validation: Internal k-fold cross-validation

Final Phase: Production v1.0 (future)
├─ Target: 15,000+ examples (1,500+ per level)
├─ Validation: Independent expert review
├─ Deployment: Docker, API, monitoring
└─ SLA: Per-class accuracy ≥85%
```

---

## 🤝 Contributing & Data Collection

**Contributions are especially welcome** for underrepresented ACTFL levels. Because the current dataset is heavily imbalanced, additional high-quality essays—especially at Advanced and Superior levels—will directly improve model reliability.

### How to Contribute

**Option 1: Submit Writing Samples**
- Anonymized essays tagged with ACTFL level
- Format: CSV with columns `essay` and `actfl_level`
- Guidelines: 100-200 words minimum, avoid copyrighted text
- Submit via: Pull request, GitHub issue, or secure link

**Option 2: Suggest Augmentation Strategies**
- Paraphrasing techniques for minority classes
- Data collection strategies from new sources
- Rebalancing algorithms
- Submit via: GitHub discussion or issue

**Option 3: Code Contributions**
- Improve evaluation metrics (per-class reports, calibration curves)
- Add new preprocessing (spell-check, grammar filtering)
- Implement ensemble models
- Add API deployment (FastAPI, Docker)
- Submit via: Pull request with description

### Data Guidelines

If submitting essays:
✅ Fully anonymized (no names, IDs, institutions)  
✅ ACTFL level labeled by expert or professional  
✅ 100+ words per sample  
✅ Diverse topics & genres  
✅ Original or properly licensed text

❌ No copyrighted text, homework, plagiarized content

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
```powershell
# In train.py, reduce batch size:
# per_device_train_batch_size=4  # was 8

# Or reduce sequence length:
# max_length=128  # was 256

# Or use gradient accumulation:
# gradient_accumulation_steps=2
```

### Missing Dataset
```
FileNotFoundError: asap_actfl_labeled.csv not found
```
**Solution:**
```powershell
# Run data preparation first:
python script.py
```

### Model Doesn't Load
```
RuntimeError: Cannot find safetensors model
```
**Solution:**
```powershell
# Verify model exists:
ls distilbert-actfl-english/
# Should contain: config.json, model.safetensors, tokenizer.json, vocab.txt

# If missing, retrain:
python train.py
```

### Poor Predictions
**Checklist:**
- Verify model file integrity (try reloading)
- Check input text (remove HTML, fix encoding)
- Consider class imbalance (see [Limitations](#️-limitations--reliability))
- For Advanced/Superior: expect low accuracy (see confusion matrix)

### Slow Inference
**Solutions:**
- Use GPU: `torch.cuda.is_available()`
- Quantize model for 4x speedup
- Batch multiple essays together

---

## 📄 License & Contact

**License:** MIT — see `LICENSE` file

**Author:** Avijit Roy  
**Website:** [avijitroy.com](https://avijitroy.com/)  
**LinkedIn:** [/in/HeyAvijitRoy](https://www.linkedin.com/in/HeyAvijitRoy/)

**Questions or issues?** Open a GitHub issue or reach out via LinkedIn.

---

## 📖 Additional Resources

- [ACTFL Proficiency Guidelines](https://www.actfl.org/guidance/actfl-proficiency-guidelines-2012)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ASAP Dataset](https://www.kaggle.com/competitions/asap-aes/data)
- [CommonLit Readability](https://www.kaggle.com/datasets/shayanfazeli/commonlit-readability-prize)
- [Gradio Documentation](https://gradio.app/docs)

---

_"Building tools to solve real problems — secure, fast, and privacy-first."_
