# Transformer Fine-Tuning Exercise
At CineStream, we ingest a huge volume of long-form user reviews for newly released films and series, far more than teams can read manually. We need a sentiment model that classifies reviews as positive or negative to power real-time dashboards, flag early signs of a release going off track, and surface strong positive reactions for marketing. We use the IMDB review dataset as a close public proxy to build and validate the fine-tuning pipeline before moving to our internal reviews.

## Objective
Fine-tune a pretrained transformer for binary sentiment classification on a small dataset and report performance. We recommend using a BERT model but you can use any model you like.

## Dataset
Using the following dataset, be sure to examine its suitability for the task on huggingface first

```python
from datasets import load_dataset
ds = load_dataset("imdb", split="train[:1000]").train_test_split(test_size=0.2)
```
You can downsample further if needed to keep training fast.

## Tasks

### 1) Load a pretrained model + tokenizer
* Use an auto model from Hugging Face (distilbert-base-uncased is a good default).
* Configure it for binary classification, consider using AutoTokenizer and AutoModelForSequenceClassification

### 2) Prepare + tokenize the dataset
* Tokenize text with truncation + padding.
* Keep max length reasonable for speed (e.g., 128–256).
* Ensure labels are correctly attached.

### 3) Fine-tune the model
We recommend using a trainer for this exercise but a custom loop is also fine. Be sure to choose sensible hyperparameters:

The training should only take a few minutes so we have enough time to complete the exercise.

### 4) Run sample predictions

Provide predictions for 3 custom texts of your choice, e.g.:
* “I loved this film.”
* “This was a waste of time.”
* “Pretty good overall, but slow in parts.”

Determine if the classification is correct and iterate until you are happy with the models performance.