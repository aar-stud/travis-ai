"""
classifer_routes.py — Intent classification for banking queries.

FIXED 1: NumPy _reconstruct compatibility shim added before pickle loads.
         Fixes "First argument must be a sub-type of ndarray" on Linux when
         .pkl was serialized on a different platform/numpy build.

FIXED 2: Added /api/query/category alias route so the frontend Dashboard
         call to POST /api/query/category resolves correctly.

FIXED 3: torch imports and model loading are fully lazy (deferred to
         first request) to avoid corrupting torch._C before main.py's
         pre-load guard runs.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import os
import re

import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

# =========================================================
# NumPy _reconstruct compatibility shim
# MUST run before any pickle.load() call on sklearn objects.
# Fixes cross-platform pickle loading when LabelEncoder.classes_
# was serialized with a different numpy C-ABI build.
# =========================================================

import numpy as np
import numpy.core.multiarray as _nmc

_original_reconstruct = _nmc._reconstruct

def _patched_reconstruct(subtype, *args, **kwargs):
    if not issubclass(subtype, np.ndarray):
        subtype = np.ndarray
    return _original_reconstruct(subtype, *args, **kwargs)

_nmc._reconstruct = _patched_reconstruct

# =========================================================
# Configuration
# =========================================================

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MAX_LEN            = 64
EMBED_DIM          = 128
N_HEADS            = 4
NUM_ENCODER_LAYERS = 2
FF_DIM             = 256
SAVE_PATH          = "best_transformer_model_90.pth"

# =========================================================
# Load vocab and label encoder at import time — safe,
# these are pure Python pickle files (now shim-protected).
# Gracefully handle missing files to avoid router registration failure.
# =========================================================

vocab = None
label_encoder = None
_missing_files = []

vocab_path = os.path.join(BASE_DIR, "vocab_90.pkl")
label_encoder_path = os.path.join(BASE_DIR, "label_encoder_90.pkl")

if os.path.exists(vocab_path):
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    except Exception as e:
        print(f"[classifier] Warning: Failed to load vocab: {e}")
        _missing_files.append("vocab_90.pkl")
else:
    _missing_files.append("vocab_90.pkl")

if os.path.exists(label_encoder_path):
    try:
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        print(f"[classifier] Warning: Failed to load label_encoder: {e}")
        _missing_files.append("label_encoder_90.pkl")
else:
    _missing_files.append("label_encoder_90.pkl")

if _missing_files:
    print(f"[classifier] Warning: Missing required files: {', '.join(_missing_files)}")

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
            self.embedding       = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.pos_encoder     = PositionalEncoding(embed_dim)
            encoder_layer        = nn.TransformerEncoderLayer(
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
    """Tokenize and pad query. Returns a torch.Tensor."""
    import torch

    tokens  = word_tokenize(query.lower())
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:MAX_LEN]]
    padded  = indices + [0] * (MAX_LEN - len(indices))
    return torch.tensor([padded], dtype=torch.long)


# AFTER — resolves device from the global without triggering a second load
def predict_category(query: str, model=None, vocab: dict = None, label_encoder=None) -> str:
    import torch

    loaded_model, device = _get_model()          # safe: returns cached instance
    _model_to_use = model if model is not None else loaded_model

    _model_to_use.eval()
    with torch.no_grad():
        input_tensor    = encode_query(query, _model_to_use if False else vocab).to(device)
        output          = _model_to_use(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([predicted_index])[0]


# =========================================================
# FastAPI Router
# =========================================================

router = APIRouter(prefix="/api", tags=["Query Classification"])


class QueryInput(BaseModel):
    query: str


async def _run_classification(input_data: QueryInput):
    query = input_data.query
    if not query:
        raise HTTPException(status_code=400, detail="Query field is required")

    if vocab is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail=f"Classifier unavailable: missing {', '.join(_missing_files)}"
        )

    try:
        # Run blocking inference in a thread so it doesn't block the event loop
        # AND wrap with a timeout so it never hangs the client indefinitely.
        result = await asyncio.wait_for(
            run_in_threadpool(_classify_sync, query),
            timeout=30.0          # 30 s hard limit
        )
        return {"category": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Classifier timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _classify_sync(query: str) -> str:
    """Pure-sync wrapper — runs in threadpool, never on the event loop."""
    model, _ = _get_model()
    return predict_category(query, model, vocab, label_encoder)

@router.post("/classify")
async def classify_query(input_data: QueryInput):
    """Primary classification endpoint."""
    return await _run_classification(input_data)


@router.post("/query/category")
async def classify_query_category(input_data: QueryInput):
    """Alias route — matches the Dashboard frontend call to /api/query/category."""
    return await _run_classification(input_data)


@router.get("/")
async def classifier_home():
    return {"message": "Category Classifier API is running!"}