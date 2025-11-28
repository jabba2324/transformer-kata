from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load small dataset with explicit balance
ds = load_dataset("imdb", split="train")
pos = ds.filter(lambda x: x["label"] == 1).select(range(500))
neg = ds.filter(lambda x: x["label"] == 0).select(range(500))
dataset = concatenate_datasets([pos, neg]).shuffle(seed=42).train_test_split(test_size=0.2)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

args = TrainingArguments(
    output_dir="out",
    # TODO: choose reasonable values
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    use_mps_device=False,
    use_cpu=True,
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# TODO: 3 example predictions
texts = [
    "I loved this film.",
    "This was a waste of time.",
    "Pretty good overall, but slow in parts."
]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
logits = model(**inputs).logits

print(logits.argmax(dim=1))