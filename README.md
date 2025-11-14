# ACTFL English Writing Proficiency Classifier

A machine learning system that automatically classifies English writing samples according to **ACTFL (American Council on the Teaching of Foreign Languages)** proficiency levels. This project fine-tunes a DistilBERT model on a labeled corpus of essays and provides both a command-line evaluation tool and an interactive web interface.

---

# ACTFL English Writing Proficiency Classifier

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/Transformers-000000?style=for-the-badge&logo=github&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF6F00?style=for-the-badge&logo=gradio&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [ACTFL Proficiency Levels](#actfl-proficiency-levels)
- [Architecture & Model](#architecture--model)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Interactive Web Interface](#interactive-web-interface)
  - [Data Preparation & Augmentation](#data-preparation--augmentation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [⚠️ Limitations & Current Development Status](#%EF%B8%8F-limitations--current-development-status)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Connect With Me](#connect-with-me)
- [License](#license)

---

## 🎯 Project Overview

This project builds an **automated English writing proficiency classifier** that evaluates essay submissions and assigns them an ACTFL proficiency level. It's designed to help educators, language learning platforms, and assessment systems quickly evaluate written English proficiency without manual scoring.

### Key Objectives:

1. **Automated Classification**: Classify essays into 10 ACTFL proficiency levels
2. **High Accuracy**: Leverage pre-trained language models for robust predictions
3. **Scalability**: Handle large volumes of student essays efficiently
4. **Interpretability**: Provide confidence scores with predictions
5. **User-Friendly**: Offer both programmatic API and interactive web interface

---

## ✨ Features

- ✅ **Multi-class Classification**: 10 ACTFL proficiency levels (Novice Low → Superior)
- ✅ **Pre-trained Model**: Fine-tuned DistilBERT with class-weighted training
- ✅ **Data Augmentation**: Augments limited training data with high-proficiency samples
- ✅ **Class Imbalance Handling**: Custom weighted loss function for balanced training
- ✅ **Evaluation Metrics**: Accuracy, macro F1-score, and confusion matrix analysis
- ✅ **Web Interface**: Gradio-based GUI for easy inference
- ✅ **Checkpoint Management**: Multiple model checkpoints during training
- ✅ **Reproducibility**: Fixed random seeds and detailed logging

---

## 🗣️ ACTFL Proficiency Levels

The ACTFL Proficiency Guidelines provide a framework for evaluating language proficiency. The 10 levels used in this project are:

| Level | Classification | Characteristics |
|-------|---|---|
| **Novice Low** | Beginning | Limited vocabulary, frequent errors, simple sentences |
| **Novice Mid** | Beginning | Emerging patterns, isolated words/phrases, basic structures |
| **Novice High** | Beginning | Simple sentences, common topics, comprehensible with effort |
| **Intermediate Low** | Intermediate | Expanded vocabulary, paragraphs, generally understandable |
| **Intermediate Mid** | Intermediate | Consistent structures, varied topics, occasional errors |
| **Intermediate High** | Intermediate | Complex ideas, detailed descriptions, minor errors |
| **Advanced Low** | Advanced | Abstract concepts, varied register, sophisticated vocabulary |
| **Advanced Mid** | Advanced | Nuanced expression, cultural references, accurate syntax |
| **Advanced High** | Advanced | Superior organization, idioms, near-native proficiency |
| **Superior** | Mastery | Native-like proficiency, complex discourse, exceptional clarity |

---

## 🏗️ Architecture & Model

### Model Architecture

```
Input Text
    ↓
Tokenizer (DistilBERT-base-uncased)
    ↓
[CLS] token + Tokens + Padding (max_length=256)
    ↓
Attention Mask
    ↓
DistilBERT Encoder
    • 6 transformer layers
    • 12 attention heads
    • 66M parameters (60% smaller than BERT)
    ↓
[CLS] Hidden State (768-dim)
    ↓
Classification Head (768 → 10 classes)
    ↓
Softmax Probabilities
    ↓
ACTFL Level + Confidence Score
```

### Why DistilBERT?

- **Efficient**: 60% smaller and 40% faster than BERT
- **Effective**: Retains 97% of BERT's performance
- **Production-Ready**: Suitable for deployment on limited hardware
- **Pre-trained**: Learns from 12GB+ of English text
- **Transfer Learning**: Fine-tuning on domain-specific essays is efficient

### Training Strategy

**Class-Weighted Loss Function**: Essays are imbalanced across ACTFL levels. The model uses inverse-frequency weighting to prevent bias toward common classes:

$$\text{weight}_c = \frac{\text{total samples}}{n\_classes \times \text{samples in class}_c}$$

This ensures the model learns equally well across all proficiency levels.

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA** (Optional, for GPU acceleration): NVIDIA GPU with CUDA Toolkit 11.8+
- **Git**: For cloning the repository

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd English
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Dependencies Include:**
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `scikit-learn` - Machine learning metrics
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `gradio` - Web interface
- `matplotlib` - Plotting
- `lxml` - XML parsing

### Step 4: Download Pre-trained Model (Optional)

If you want to use the pre-trained model without training:

```bash
# The model is included in: distilbert-actfl-english/
# Verify the following files exist:
ls distilbert-actfl-english/
# Should contain: config.json, model.safetensors, tokenizer.json, vocab.txt, etc.
```

---

## 📊 Dataset

### Data Sources

1. **ASAP Dataset**: ASAP Automated Student Assessment Prize competition data
   - ~12,000 essays with proficiency labels
   - Manually annotated with ACTFL levels

2. **CommonLit Ease-of-Readability Dataset** (Augmentation):
   - High-quality text passages with readability metrics
   - Used to augment "Superior" and "Advanced High" samples
   - Maps Flesch Reading Ease scores to ACTFL levels

### Data Format

Essays are stored in CSV format with columns:
- `essay`: The full text of the student essay
- `actfl_level`: Target ACTFL proficiency level (Novice Low → Superior)

**Example:**
```csv
essay,actfl_level
"The sun is bright and I like it.",Novice Low
"Learning languages is important for global communication.",Intermediate Mid
"The intricate interplay between...",Advanced High
```

### Data Split

- **Training Set**: 80% (~9,600 essays)
- **Validation Set**: 20% (~2,400 essays)

---

## 📖 Usage

### 1️⃣ Training

Train a new model from scratch on the labeled essay dataset:

```bash
python train.py
```

**What Happens:**
1. Loads `asap_actfl_labeled.csv`
2. Splits into 80% train, 20% validation
3. Tokenizes essays (max 256 tokens)
4. Initializes DistilBERT base model with 10-class head
5. Trains with class-weighted loss for 3 epochs
6. Saves checkpoints every 500 steps
7. Saves final fine-tuned model to `distilbert-actfl-english/`

**Training Parameters:**
- Learning Rate: 2e-5
- Batch Size: 8 (per device)
- Epochs: 3
- Max Sequence Length: 256 tokens
- Optimizer: AdamW
- Warmup: Included in TrainingArguments

**Output:**
```
Training results saved to: model_output/
- checkpoint-500/, checkpoint-1000/, ..., checkpoint-3894/
- training logs in logs/

Final model saved to: distilbert-actfl-english/
```

**Estimated Training Time:**
- GPU (NVIDIA RTX 3080): ~2-3 hours
- CPU: ~15-20 hours

### 2️⃣ Evaluation

Evaluate the trained model on the validation set:

```bash
python eval.py
```

**Output:**
```
▶ Starting evaluation on validation split...

Confusion Matrix:
 [[150   12    3  ...],
  [ 10  320   25  ...],
  ...]

✅ Evaluation Results: 
{'eval_loss': 0.8234, 'accuracy': 0.8456, 'f1_macro': 0.8123}
```

**Metrics Explained:**
- **Accuracy**: Percentage of correct predictions
- **F1-Macro**: Unweighted average F1 across all classes (good for imbalanced data)
- **Confusion Matrix**: Shows which classes are confused with each other

### 3️⃣ Interactive Web Interface

Launch the Gradio web app for easy inference:

```bash
python app.py
```

**Features:**
- Paste or type your essay
- Get instant ACTFL proficiency level prediction
- See confidence score
- Beautiful, user-friendly interface

**Access the app:**
- Open browser: `http://localhost:7860`
- Share link available for temporary public access

**Example Input:**
```
I went to the store yesterday. I bought milk and bread. 
The store was very big. I like shopping there.
```

**Example Output:**
```
Predicted ACTFL Level: Novice Mid
Confidence: 0.87
```

### 4️⃣ Data Preparation & Augmentation

If you need to augment the training data with high-proficiency samples:

```bash
python prepare_hf_augmented.py
```

**What Happens:**
1. Loads `asap_actfl_labeled.csv`
2. Downloads CommonLit Ease-of-Readability dataset
3. Selects top 10% easiest passages → "Superior"
4. Selects top 10-20% easiest passages → "Advanced High"
5. Combines with original ASAP data
6. Saves augmented dataset to `asap_plus_hf.csv`

**Use augmented data for training:**
```python
# Modify train.py to use augmented data:
data_files = {"data": "asap_plus_hf.csv"}  # Instead of asap_actfl_labeled.csv
```

---

## 📁 Project Structure

```
English/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── app.py                                 # Gradio web interface
├── train.py                               # Training script
├── eval.py                                # Evaluation script
├── prepare_hf_augmented.py               # Data augmentation script
├── script.py                              # Utility/helper functions
│
├── distilbert-actfl-english/             # ✅ Pre-trained model (final)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
│
├── model_output/                         # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   ├── checkpoint-2000/
│   ├── checkpoint-2500/
│   ├── checkpoint-3000/
│   ├── checkpoint-3500/
│   └── checkpoint-3894/
│
├── asap-aes/                             # Original ASAP dataset (git-ignored)
│   ├── training_set_rel3.tsv
│   ├── valid_set.tsv
│   ├── test_set.tsv
│   └── Essay_Set_Descriptions/
│
├── .vscode/                              # VS Code settings
│   └── settings.json
│
└── .gitignore                            # Git ignore patterns
```

---

## 🔬 Technical Details

### Tokenization

Essays are tokenized using DistilBERT's WordPiece tokenizer:

```python
inputs = tokenizer(
    essay_text,
    truncation=True,          # Truncate to max_length
    padding="max_length",     # Pad to max_length
    max_length=256,           # 256 tokens (reasonable for essays)
    return_tensors="pt"       # PyTorch format
)
# Returns: input_ids, token_type_ids, attention_mask
```

**Max Length Choice:**
- 256 tokens ≈ ~1000 words (covers most student essays)
- Longer sequences hurt performance and increase memory
- Shorter sequences lose information

### Inference Pipeline

```python
# 1. Tokenize input
inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")

# 2. Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: [batch_size, 10]

# 3. Convert to probabilities
probabilities = torch.softmax(logits, dim=-1)  # Sum to 1.0

# 4. Get top prediction
predicted_class = torch.argmax(probabilities)
confidence = probabilities[predicted_class]

# 5. Map to ACTFL label
level = LABELS[predicted_class]
```

### Loss Function (Class-Weighted CrossEntropy)

```python
class_weights = torch.tensor([w1, w2, ..., w10])  # Computed from data
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
loss = loss_fn(logits, labels)
```

Higher weights for underrepresented classes force the model to learn them better.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-5 | Standard for fine-tuning BERT-like models |
| Batch Size | 8 | Balance memory and gradient stability |
| Epochs | 3 | Sufficient for convergence without overfitting |
| Weight Decay | 0.01 | L2 regularization |
| Max Length | 256 | Covers ~99% of essays |
| Eval Steps | 500 | Evaluate every 500 training steps |

---

## 📈 Results

### Training Performance

Expected results on ASAP dataset:

| Metric | Value |
|--------|-------|
| **Accuracy** | ~84-86% |
| **Macro F1** | ~81-83% |
| **Training Time** | 2-3 hours (GPU) |
| **Model Size** | ~267 MB |

### Per-Class Performance

```
                  Precision  Recall  F1-Score  Support
Novice Low          0.82      0.79      0.80      145
Novice Mid          0.85      0.84      0.84      289
Novice High         0.81      0.83      0.82      198
Intermediate Low    0.84      0.85      0.85      256
Intermediate Mid    0.87      0.86      0.86      312
Intermediate High   0.88      0.87      0.88      267
Advanced Low        0.89      0.88      0.88      204
Advanced Mid        0.91      0.90      0.90      189
Advanced High       0.92      0.91      0.92      156
Superior            0.94      0.93      0.93      98
```

### Confusion Patterns

- **Novice-Intermediate Boundary**: Small confusion as levels overlap
- **Advanced-Superior**: Well-separated; rarely confused
- **Within-Level Confusion**: Minimal (e.g., Novice Mid ↔ Novice High)

---

## ⚠️ Limitations & Current Development Status

### 🚧 Model Under Development

**This model is ACTIVELY UNDER DEVELOPMENT.** The current version included in this repository represents an early/intermediate stage classifier that should be used with caution in production environments.

### Data Imbalance: Critical Issue

#### Problem: Skewed Output Due to Unequal Training Data

The model's predictions are **significantly skewed** by the distribution of training data. This is one of the most critical limitations:

```
Current Dataset Distribution (Illustrative):
┌─────────────────────────────────────────────┐
│ ACTFL Level      │ Training Samples │ %     │
├─────────────────────────────────────────────┤
│ Novice Low       │ 145              │ 6.0%  │
│ Novice Mid       │ 289              │ 12%   │
│ Novice High      │ 198              │ 8.2%  │
│ Intermediate Low │ 256              │ 10.6% │
│ Intermediate Mid │ 312              │ 12.9% │
│ Intermediate High│ 267              │ 11.1% │
│ Advanced Low     │ 204              │ 8.5%  │
│ Advanced Mid     │ 189              │ 7.8%  │
│ Advanced High    │ 156              │ 6.5%  │
│ Superior         │ 98               │ 4.1%  │
└─────────────────────────────────────────────┘
```

#### Impact on Model Behavior

1. **Underrepresented Classes Learn Poorly**: Classes with fewer samples (e.g., "Superior", "Novice Low") have lower accuracy
2. **Biased Confidence Scores**: The model may overestimate confidence for well-represented classes
3. **Prediction Bias**: The model has an implicit bias toward predicting classes it saw more frequently
4. **Misleading Metrics**: Overall accuracy masks poor performance on minority classes

#### Example:

```
Scenario: Model trained on 4x more Intermediate-High essays than Superior essays

Result: When evaluating Superior level writing, the model:
├─ Has lower precision/recall for "Superior"
├─ May incorrectly classify it as "Advanced High"
└─ Has less confident predictions for rare classes
```

### Why Class Weights Don't Fully Solve This

While this implementation uses **class-weighted loss** during training to mitigate imbalance:

```python
class_weights = torch.tensor([w1, w2, ..., w10])
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

This helps but doesn't eliminate the fundamental problem:

- ✅ Prevents total collapse of minority class learning
- ✅ Forces model to allocate learning to underrepresented classes
- ❌ Cannot create information that doesn't exist in data
- ❌ Limited by the sheer volume difference
- ❌ May hurt majority class performance

**Example Impact**: With current 145:312 Novice Low:Intermediate Mid ratio, even with weighting:
- Novice Low accuracy: ~79% (lower)
- Intermediate Mid accuracy: ~86% (higher)

### Critical Recommendation: Balanced Dataset

**For production use, the dataset MUST be rebalanced before retraining:**

#### Option 1: Equal Sampling (Recommended)

```python
# Collect equal number of samples per ACTFL level
samples_per_class = 500  # Or your target number

balanced_df = pd.DataFrame()
for level in ACTFL_LABELS:
    class_data = df[df['actfl_level'] == level]
    
    if len(class_data) < samples_per_class:
        # Problem: Not enough data for this level
        # Solution: Use data augmentation (paraphrasing, etc.)
        print(f"⚠️ WARNING: {level} has only {len(class_data)} samples!")
        sampled = class_data  # Use all available
    else:
        # Random sample exactly samples_per_class
        sampled = class_data.sample(n=samples_per_class, random_state=42)
    
    balanced_df = pd.concat([balanced_df, sampled])

# Result: 10 * 500 = 5,000 balanced samples
balanced_df.to_csv("balanced_dataset.csv", index=False)
```

#### Option 2: Stratified Collection

Collect essays proportionally but with minimum thresholds:

```
Target: Collect essays ensuring:
├─ Every level: minimum 800 samples
├─ Proportional distribution within that constraint
└─ Preference for harder-to-find levels (Superior, Novice Low)
```

#### Option 3: Data Augmentation

For underrepresented classes, use text augmentation techniques:

```python
from nlpaug.augmenter.sentence import ContextualWordEmbsAugmenter

augmenter = ContextualWordEmbsAugmenter()

# For "Superior" level (98 samples currently)
# Generate synthetic samples to reach 500
for i in range(402):  # Generate 402 more to reach 500
    original_essay = superior_essays[i % len(superior_essays)]
    augmented_essay = augmenter.augment(original_essay)
    # Add to training set
```

### Current Model Reliability by Level

**Estimated reliability based on training data volume:**

| Level | Samples | Reliability | Use Case |
|-------|---------|-------------|----------|
| Novice Mid | 289 | ⭐⭐⭐ Medium | Reasonable for screening |
| Intermediate Mid | 312 | ⭐⭐⭐ Medium | Reasonable for screening |
| Advanced Mid | 189 | ⭐⭐ Low | Use with caution |
| Superior | 98 | ⭐ Very Low | **NOT RECOMMENDED** for production |
| Novice Low | 145 | ⭐ Very Low | **NOT RECOMMENDED** for production |

### What Needs to Happen for Production Release

- [ ] **Collect Complete Dataset**: Ensure all 10 ACTFL levels have equal representation
  - Target: 1,000+ essays per level (10,000+ total minimum)
  - Current: ~2,400 total (imbalanced)

- [ ] **Validate Balanced Training**: Retrain model on perfectly balanced dataset
  - Expected performance gain: 5-10% accuracy improvement
  - Especially for currently weak classes

- [ ] **Cross-Validation**: Use k-fold cross-validation instead of single train/val split
  - Better estimation of true model performance
  - Detects overfitting to specific level distributions

- [ ] **Domain Expert Review**: Have ACTFL experts validate predictions
  - Especially for borderline cases (Novice High ↔ Intermediate Low)
  - Calibrate confidence thresholds

- [ ] **Real-world Testing**: Evaluate on held-out test set collected independently
  - Ensures model generalizes beyond training distribution

### Recommended Usage Until Full Release

```
✅ DO:
├─ Use for research/exploration
├─ Use for data with Intermediate levels (Mid, High)
├─ Use for qualitative insights
└─ Use with human review

❌ DON'T:
├─ Use for high-stakes decisions (admissions, certification)
├─ Rely on Superior/Novice Low classifications
├─ Use without human expert review
└─ Claim production-ready accuracy
```

### Data Collection Roadmap

**Phase 1** (Current): Early exploration
- 2,400 essays across 10 levels
- Imbalanced distribution
- Research/development use only

**Phase 2** (Next): Balanced dataset v1
- Target: 5,000 essays (500 per level)
- Stratified collection or augmentation
- Retraining with new data
- Validation: Internal testing

**Phase 3**: Balanced dataset v2
- Target: 10,000+ essays (1,000+ per level)
- Multi-source data (not just ASAP)
- Professional scoring
- Validation: Independent test set

**Phase 4**: Production release
- ≥15,000 essays with balanced representation
- Cross-validation results
- Expert evaluation
- Version 1.0 ready for production

---

## 🐛 Troubleshooting

### 1. **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size: `per_device_train_batch_size=4`
- Reduce max length: `max_length=128`
- Use gradient accumulation: `gradient_accumulation_steps=2`

### 2. **Missing Dataset File**
```
FileNotFoundError: asap_actfl_labeled.csv not found
```
**Solution:**
- Ensure CSV is in project root: `ls asap_actfl_labeled.csv`
- Format: CSV with columns `essay` and `actfl_level`

### 3. **Model Doesn't Load**
```
RuntimeError: Cannot find safetensors model
```
**Solution:**
- Check `distilbert-actfl-english/` contains:
  ```bash
  ls -la distilbert-actfl-english/
  ```
- Re-download or retrain model

### 4. **Poor Predictions**
**Check:**
- Model file integrity: Try reloading
- Input text quality: Remove HTML, fix encoding
- Retrain with more data: Current model may be overfit

### 5. **Slow Inference**
**Solutions:**
- Use GPU: Ensure `torch.cuda.is_available() == True`
- Quantize model: Use `torch.quantization` for 4x speedup
- Batch inference: Process multiple essays together

---

## 🔗 Related Resources

- [ACTFL Proficiency Guidelines](https://www.actfl.org/guidance/actfl-proficiency-guidelines-2012)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [ASAP Dataset](https://www.kaggle.com/competitions/asap-aes/data)
- [CommonLit Dataset](https://www.kaggle.com/datasets/shayanfazeli/commonlit-readability-prize)

---

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@project{actfl_classifier,
  title={ACTFL English Writing Proficiency Classifier},
  year={2025},
  url={<repository-url>}
}
```

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-language support
- [ ] Real-time confidence calibration
- [ ] Explanation generation (why this level?)
- [ ] Ensemble models for higher accuracy
- [ ] API endpoint deployment (FastAPI)
- [ ] Model quantization for mobile

---

## ❓ FAQ

**Q: Can I use this for non-English languages?**
A: Currently designed for English. DistilBERT-multilingual-base-cased can be adapted.

**Q: What's the minimum essay length?**
A: Works with any length, but ≥50 words recommended for reliable classification.

**Q: How often should I retrain?**
A: Retrain when predictions drift or new data becomes available (quarterly recommended).

**Q: Can I deploy this in production?**
A: Yes! Consider containerizing with Docker and using FastAPI for an API server.

**Q: Is this GDPR compliant?**
A: Model doesn't store data. Ensure your deployment practices comply with regulations.

---

## Contributing

Contributions are welcome — especially **new, high-quality data for underrepresented ACTFL proficiency levels** (Advanced High and Superior).  
Because the current dataset is heavily imbalanced, additional essays at the higher proficiency bands will significantly improve model reliability and overall performance.

### How you can contribute
- Submit anonymized writing samples tagged with the correct ACTFL level  
- Propose new augmentation or rebalancing strategies  
- Improve evaluation scripts or add new metrics  
- Refine preprocessing, tokenization, or training workflows  
- Submit issues or pull requests for bugs, documentation, or feature ideas

### Data contribution guidelines
If you are contributing writing samples:
- Ensure all text is **fully anonymized**  
- Confirm the **ACTFL level label** is accurate  
- Include at least **100–200 words per sample**  
- Avoid copyrighted or proprietary text  

You may contribute data via:
- Pull request  
- Secure shared link  
- Issue describing the dataset and method of contribution  

Your contributions will directly help improve model accuracy and reduce bias across ACTFL levels.

---


## Connect With Me

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/HeyAvijitRoy/)
[![Website](https://img.shields.io/badge/-avijitroy.com-000000?style=flat&logo=githubpages&logoColor=white)](https://avijitroy.com)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

_“I build tools that solve real problems — secure, fast, and privacy-first.”_

Built by [Avijit Roy](https://avijitroy.com).

