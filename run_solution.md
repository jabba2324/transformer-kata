# How to Run the Solution

## Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Script

Execute the solution:
```bash
python solution.py
```

## What It Does

1. **Loads Data**: Downloads 1000 IMDB reviews (500 positive, 500 negative)
2. **Tokenizes**: Processes text using DistilBERT tokenizer
3. **Fine-tunes**: Trains the model for 3 epochs on CPU
4. **Evaluates**: Reports accuracy and F1 score
5. **Predicts**: Tests on 3 sample reviews

## Expected Output

- Training progress with loss/accuracy metrics
- Final evaluation scores
- Predictions for the 3 test sentences (0=negative, 1=positive)

## Runtime

Approximately 5-10 minutes on CPU depending on your machine.