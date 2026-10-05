# 🎬 SentimentScope

A transformer-based sentiment classifier for movie reviews, built from scratch in PyTorch and trained on the IMDB dataset. Includes a Streamlit app for live inference.

**Live demo:** 

## Overview

SentimentScope classifies a movie review as **positive** or **negative**. The project started as a scenario for a fictional company, CineScope, that wants to understand user sentiment to improve its recommendation system.

Instead of fine-tuning a pretrained model, the transformer encoder is implemented and trained from scratch. Only the `bert-base-uncased` tokenizer is borrowed from Hugging Face.

## Results

| Split      | Accuracy |
|------------|----------|
| Validation | 84.12%   |
| Test       | 81.92%   |

Trained for 3 epochs. Validation accuracy improved each epoch (80.16% → 82.92% → 84.12%).

## Model Architecture

| Component          | Value                                   |
|--------------------|-----------------------------------------|
| Tokenizer          | `bert-base-uncased` (subword)           |
| Max sequence length| 248 tokens                              |
| Embedding size     | 248                                     |
| Transformer blocks | 8                                       |
| Attention heads    | 4 (head size 32)                        |
| Feed-forward       | 4× expansion, GELU                      |
| Dropout            | 0.1                                     |
| Pooling            | Mean over token embeddings              |
| Output             | Linear layer → 2 classes                |

Each block uses pre-layer-norm residual connections around multi-head self-attention and a feed-forward network. Token and learned positional embeddings are summed at the input.

## Training Setup

- **Dataset:** [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (25,000 train / 25,000 test)
- **Split:** 90% train (22,500) / 10% validation (2,500), shuffled with a fixed seed
- **Loss:** Cross-entropy
- **Optimizer:** AdamW, learning rate 3e-4
- **Batch size:** 16
- **Epochs:** 3
- **Hardware:** single GPU

## Project Structure

```
├── SentimentScope.ipynb   # Data exploration, training, evaluation
├── model_def.py           # Model architecture, config, and weight-loading helper
├── app.py                 # Streamlit inference app
├── model.pt               # Trained weights (stored with Git LFS)
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone and install

```bash
git lfs install
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Use the model in code

```python
import torch
from model_def import load_model, tokenizer, MAX_LENGTH

model = load_model("model.pt")
device = next(model.parameters()).device

enc = tokenizer(
    "I loved this movie!",
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
pred = model(enc["input_ids"].to(device)).argmax(dim=1).item()
print("positive" if pred == 1 else "negative")
```

## Limitations

- Reviews longer than 248 tokens are truncated, so sentiment expressed late in a long review may be missed.
- The model only predicts positive or negative. There is no neutral class, so mixed reviews get forced into one label.
- It was trained only on IMDB movie reviews and may perform worse on other text such as product reviews or tweets.
- The attention layers keep a causal mask from the original generative design, so each token only attends to earlier tokens. Removing it is a natural experiment to try.

## Ideas for Improvement

- Train for more epochs and tune the learning rate with a scheduler
- Increase embedding size or the number of layers
- Remove the causal attention mask, since classification doesn't need it
- Use an attention mask so padding tokens are ignored in mean pooling
- Compare against a fine-tuned pretrained BERT as a baseline

## Acknowledgements

- Dataset: Maas et al., *Learning Word Vectors for Sentiment Analysis* (ACL 2011)
- Tokenizer: Hugging Face `bert-base-uncased`

## Author

Built by Eric, Computer Science student (AI/ML) at Dedan Kimathi University of Technology.