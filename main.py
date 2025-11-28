from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load small dataset
dataset = load_dataset("imdb", split="train[:1000]").train_test_split(test_size=0.2)

model_name = "distilbert-base-uncased"
# TODO: Load tokenizer

def tokenize_fn(batch):
    # TODO: Tokenize the text
    pass

# TODO: Apply tokenization to dataset

# TODO: Load model for sequence classification

def compute_metrics(p):
    # TODO: Compute accuracy and F1 score
    return {}

args = TrainingArguments(
    output_dir="out",
    # TODO: choose reasonable values
)

# TODO: Initialize Trainer

# TODO: Train and Evaluate

# TODO: 3 example predictions
texts = [
    "I loved this film.",
    "This was a waste of time.",
    "Pretty good overall, but slow in parts."
]