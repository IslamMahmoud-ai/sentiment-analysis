"""
sentiment_pipeline.py — Real-Time Sentiment Analysis Platform
BERT fine-tuning + Azure Functions deployment
pip install transformers torch datasets azure-functions
"""

import torch
from transformers import (BertTokenizer, BertForSequenceClassification,
                           Trainer, TrainingArguments)
from datasets import load_dataset
import json

LABELS     = ["negative", "neutral", "positive"]
MODEL_NAME = "bert-base-uncased"
SAVE_DIR   = "./bert-sentiment/best"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# ─── Tokenize ─────────────────────────────────────────────────────
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length",
                     truncation=True, max_length=128)


# ─── Fine-Tune ────────────────────────────────────────────────────
def fine_tune():
    dataset = load_dataset("tweet_eval", "sentiment")
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids","attention_mask","label"])

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3)

    args = TrainingArguments(
        output_dir="./bert-sentiment",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ Model saved → {SAVE_DIR}")


# ─── Inference ────────────────────────────────────────────────────
_model, _tok = None, None

def _load():
    global _model, _tok
    if _model is None:
        _tok   = BertTokenizer.from_pretrained(SAVE_DIR)
        _model = BertForSequenceClassification.from_pretrained(SAVE_DIR)
        _model.eval()


def predict_sentiment(text: str) -> dict:
    _load()
    inputs = _tok(text, return_tensors="pt",
                  truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx   = int(probs.argmax())
    return {
        "label":      LABELS[idx],
        "confidence": round(probs[idx].item(), 4),
        "scores":     dict(zip(LABELS, [round(p.item(), 4) for p in probs])),
    }


def predict_batch(texts: list[str]) -> list[dict]:
    return [predict_sentiment(t) for t in texts]


# ─── Azure Function HTTP Trigger ──────────────────────────────────
try:
    import azure.functions as func

    def main(req: func.HttpRequest) -> func.HttpResponse:
        try:
            body = req.get_json()
        except ValueError:
            return func.HttpResponse("Invalid JSON", status_code=400)

        texts = body.get("texts") or ([body.get("text")] if body.get("text") else None)
        if not texts:
            return func.HttpResponse("Provide 'text' or 'texts'", status_code=400)

        results = predict_batch(texts)
        return func.HttpResponse(json.dumps(results),
                                 mimetype="application/json")
except ImportError:
    pass   # running locally without azure SDK


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "I love this product, it works great!"
    print(predict_sentiment(text))
