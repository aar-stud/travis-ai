"""
classifer_routes.py — Intent classification for banking queries.

FIXED: torch imports and model loading are now fully lazy (deferred to
first request). Previously, module-level torch imports and torch.load()
were corrupting torch._C before main.py's pre-load guard could run,
breaking every other ML service in the process.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
import pickle
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Configuration
# =========================================================

MAX_LEN        = 64
EMBED_DIM      = 128
N_HEADS        = 4
NUM_ENCODER_LAYERS = 2
FF_DIM         = 256
SAVE_PATH      = "best_transformer_model_90.pth"

# =========================================================
# Load vocab and label encoder at import time — safe,
# these are pure Python pickle files, no C extensions.
# =========================================================

with open(os.path.join(BASE_DIR, "vocab_90.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder_90.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# =========================================================
# Lazy globals — torch and model loaded on first request
# =========================================================

_model  = None
_device = None


def _get_model():
    """Load the classifier model once, lazily, on first call."""
    global _model, _device

    if _model is not None:
        return _model, _device

    import torch
    import torch.nn as nn

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model classes (defined here so torch is already imported) ----

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=MAX_LEN):
            super().__init__()
            pe       = torch.zeros(max_len, d_model)
            pos      = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2)
                * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            self.pe = pe.unsqueeze(0)

        def forward(self, x):
            return x + self.pe[:, : x.size(1)].to(x.device)

    class TransformerClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
                     num_layers, num_classes):
            super().__init__()
            self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.pos_encoder = PositionalEncoding(embed_dim)
            encoder_layer   = nn.TransformerEncoderLayer(
                embed_dim, num_heads, ff_dim, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers
            )
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)
            return self.fc(x)

    # ---- Load weights ----

    vocab_size  = len(vocab)
    num_classes = len(label_encoder.classes_)

    m = TransformerClassifier(
        vocab_size, EMBED_DIM, N_HEADS, FF_DIM, NUM_ENCODER_LAYERS, num_classes
    ).to(_device)
    m.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, SAVE_PATH),
            map_location=_device,
        )
    )
    m.eval()
    _model = m

    return _model, _device


# =========================================================
# Pure-python helpers (no torch dependency)
# =========================================================

def word_tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())


def encode_query(query: str, vocab: dict):
    """Tokenize and pad query. Returns a torch.Tensor on first real call."""
    import torch

    tokens  = word_tokenize(query.lower())
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:MAX_LEN]]
    padded  = indices + [0] * (MAX_LEN - len(indices))
    return torch.tensor([padded], dtype=torch.long)


def predict_category(query: str, model, vocab: dict, label_encoder) -> str:
    import torch

    _, device = _get_model()          # ensures _device is set
    model.eval()
    with torch.no_grad():
        input_tensor    = encode_query(query, vocab).to(device)
        output          = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([predicted_index])[0]


# =========================================================
# FastAPI Router
# =========================================================

router = APIRouter(prefix="/api", tags=["Query Classification"])


class QueryInput(BaseModel):
    query: str


@router.post("/classify")
async def classify_query(input_data: QueryInput):
    query = input_data.query
    if not query:
        raise HTTPException(status_code=400, detail="Query field is required")
    try:
        model, _ = _get_model()
        category = predict_category(query, model, vocab, label_encoder)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def classifier_home():
    return {"message": "Category Classifier API is running!"}