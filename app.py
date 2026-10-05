"""
app.py
Streamlit demo for SentimentScope: transformer-based IMDB sentiment classifier.

Run with:
    streamlit run app.py

Requires model_def.py and the trained weights (model.pt) in the same folder.
"""
import os

import streamlit as st
import torch

from model_def import MAX_LENGTH, load_model, tokenizer

WEIGHTS_PATH = "model.pth"  # change to "model.pth" if you saved it that way
LABELS = {0: "Negative", 1: "Positive"}

EXAMPLES = {
    "Positive example": "The movie was a rollercoaster of emotions, and I loved every moment of it! "
                        "The acting was superb and the story kept me hooked until the end.",
    "Negative example": "The plot was predictable, and the acting was subpar. "
                        "A waste of time, I wanted to walk out halfway through.",
    "Mixed example": "The visuals were gorgeous, but the story dragged and the ending felt rushed.",
}

st.set_page_config(page_title="SentimentScope", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(WEIGHTS_PATH, device=device)
    return model, device


def predict(text, model, device):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred = int(torch.argmax(probs).item())
    n_tokens = int(enc["attention_mask"].sum().item())
    return pred, probs, n_tokens


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🎬 SentimentScope")
st.caption(
    "A transformer trained from scratch in PyTorch to classify IMDB movie reviews "
    "as positive or negative."
)

if not os.path.exists(WEIGHTS_PATH):
    st.error(
        f"Could not find `{WEIGHTS_PATH}`. Save your trained weights first with "
        f"`torch.save(model.state_dict(), '{WEIGHTS_PATH}')` and place the file next to app.py."
    )
    st.stop()

model, device = get_model()

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

st.subheader("Try it out")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.review_text = text

review = st.text_area(
    "Enter a movie review",
    key="review_text",
    height=180,
    placeholder="Type or paste a movie review here...",
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        pred, probs, n_tokens = predict(review, model, device)
        label = LABELS[pred]
        confidence = probs[pred].item() * 100

        if pred == 1:
            st.success(f"😊 **{label}** ({confidence:.1f}% confidence)")
        else:
            st.error(f"😞 **{label}** ({confidence:.1f}% confidence)")

        st.write("**Class probabilities**")
        st.progress(float(probs[1]), text=f"Positive: {probs[1].item() * 100:.1f}%")
        st.progress(float(probs[0]), text=f"Negative: {probs[0].item() * 100:.1f}%")

        if n_tokens >= MAX_LENGTH:
            st.info(
                f"Your review was longer than {MAX_LENGTH} tokens, so the end was truncated "
                "before classification."
            )

# ----------------------------------------------------------------------------
# About section
# ----------------------------------------------------------------------------
with st.expander("About this model"):
    st.markdown(
        f"""
- **Architecture:** 8-layer transformer, 4 attention heads, {MAX_LENGTH}-dim embeddings
- **Tokenizer:** `bert-base-uncased` (subword tokenization)
- **Pooling:** mean pooling over token embeddings, then a linear classification head
- **Training data:** IMDB reviews (22,500 train / 2,500 validation / 25,000 test)
- **Results:** about 84% validation accuracy and 82% test accuracy
- **Max input length:** {MAX_LENGTH} tokens (longer reviews are truncated)
- **Running on:** `{device}`
"""
    )