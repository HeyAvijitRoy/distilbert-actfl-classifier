"""
app.py

Small Gradio interface for inference using saved model 'distilbert-actfl-english'.
Paste text and get predicted ACTFL label + probabilities.

Run:
    python app.py
Then open the local Gradio URL printed in the terminal.
"""

import gradio as gr
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_DIR = "distilbert-actfl-english"
LABELS = [
    "Novice Low", "Novice Mid", "Novice High",
    "Intermediate Low", "Intermediate Mid", "Intermediate High",
    "Advanced Low", "Advanced Mid", "Advanced High",
    "Superior"
]

# Load once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict(text: str):
    if not text or text.strip() == "":
        return "No text provided", {}, 0.0

    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        # Build readable probs dict
        probs_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return pred_label, probs_dict, confidence


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, placeholder="Paste student text here..."),
    outputs=[
        gr.Textbox(label="Predicted ACTFL Level"),
        gr.JSON(label="Probabilities by Level"),
        gr.Label(num_top_classes=1, label="Confidence")
    ],
    title="ACTFL English Classifier (DistilBERT)",
    description="Paste student text and get predicted ACTFL level and confidence. Model: distilbert-actfl-english"
)

if __name__ == "__main__":
    demo.launch(share=False)
