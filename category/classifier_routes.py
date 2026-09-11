"""
classifier_routes.py — Intent classification for banking queries.

FIXED 1: Model dimensions (vocab_size, num_classes) are now read directly
         from the checkpoint weights instead of being inferred from the
         .pkl files. The checkpoint is the ground truth — it tells us:
           embedding.weight shape[0] = actual vocab_size  (3322)
           fc.weight        shape[0] = actual num_classes (95)

FIXED 2: numpy _reconstruct shim patches both legacy and new numpy paths
         so label_encoder_90.pkl unpickles correctly on Linux.

FIXED 3: /api/query/category alias for the frontend Dashboard.

FIXED 4: All torch imports and model loading are fully lazy.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import os
import re
import asyncio
from fastapi.concurrency import run_in_threadpool

# =========================================================
# numpy _reconstruct compatibility shim
# Must run before any pickle.load() on sklearn objects.
# Patches BOTH numpy paths — which one sklearn hits depends
# on the numpy internal layout (changed in numpy 1.25).
# =========================================================

import numpy as np


def _make_safe_reconstruct(original_fn):
    def _safe(subtype, *args, **kwargs):
        if not (isinstance(subtype, type) and issubclass(subtype, np.ndarray)):
            subtype = np.ndarray
        return original_fn(subtype, *args, **kwargs)
    return _safe


try:
    import numpy.core.multiarray as _nmc_legacy
    _nmc_legacy._reconstruct = _make_safe_reconstruct(_nmc_legacy._reconstruct)
    # ALIAS for pickle: if pickle looks for numpy._core, give it numpy.core
    import sys
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.multiarray'] = _nmc_legacy
except Exception as _e:
    print(f"[classifier] numpy.core shim skipped: {_e}")

try:
    import numpy._core.multiarray as _nmc_new
    _nmc_new._reconstruct = _make_safe_reconstruct(_nmc_new._reconstruct)
except Exception as _e:
    print(f"[classifier] numpy._core shim skipped: {_e}")

# =========================================================
# Configuration — architecture hyperparameters only.
# vocab_size and num_classes are read from the checkpoint.
# =========================================================

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MAX_LEN            = 64
EMBED_DIM          = 128
N_HEADS            = 4
NUM_ENCODER_LAYERS = 2
FF_DIM             = 256
SAVE_PATH          = "best_transformer_model_90.pth"

# =========================================================
# Load vocab and label_encoder at import time.
# Graceful failure: router still registers, bad requests → 503.
# =========================================================

vocab          = None
label_encoder  = None
_missing_files = []

_vocab_path = os.path.join(BASE_DIR, "vocab_90.pkl")
_le_path    = os.path.join(BASE_DIR, "label_encoder_90.pkl")

if os.path.exists(_vocab_path):
    try:
        with open(_vocab_path, "rb") as _f:
            vocab = pickle.load(_f)
        print(f"[classifier] vocab loaded — {len(vocab)} tokens")
    except Exception as _e:
        print(f"[classifier] vocab load FAILED: {_e}")
        _missing_files.append("vocab_90.pkl")
else:
    print(f"[classifier] vocab_90.pkl not found at {_vocab_path}")
    _missing_files.append("vocab_90.pkl")

if os.path.exists(_le_path):
    try:
        with open(_le_path, "rb") as _f:
            label_encoder = pickle.load(_f)
        print(f"[classifier] label_encoder loaded — {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")
    except Exception as _e:
        print(f"[classifier] label_encoder load FAILED: {_e}")
        _missing_files.append("label_encoder_90.pkl")
else:
    print(f"[classifier] label_encoder_90.pkl not found at {_le_path}")
    _missing_files.append("label_encoder_90.pkl")

# =========================================================
# Lazy globals
# =========================================================

_model  = None
_device = None


def _get_model():
    global _model, _device

    if _model is not None:
        return _model, _device

    import torch
    import torch.nn as nn

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Read true dimensions from the checkpoint ----
    # The checkpoint is the ground truth for vocab_size and num_classes.
    # Do NOT infer these from the .pkl files — they can differ.
    checkpoint_path = os.path.join(BASE_DIR, SAVE_PATH)
    state_dict      = torch.load(checkpoint_path, map_location=_device)

    vocab_size  = state_dict["embedding.weight"].shape[0]   # e.g. 3322
    num_classes = state_dict["fc.weight"].shape[0]           # e.g. 95

    print(f"[classifier] checkpoint dims — vocab_size={vocab_size}, num_classes={num_classes}")

    # ---- Model ----

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
        def __init__(self, vocab_size, embed_dim, num_heads,
                     ff_dim, num_layers, num_classes):
            super().__init__()
            self.embedding           = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.pos_encoder         = PositionalEncoding(embed_dim)
            encoder_layer            = nn.TransformerEncoderLayer(
                embed_dim, num_heads, ff_dim, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc                  = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)
            return self.fc(x)

    m = TransformerClassifier(
        vocab_size, EMBED_DIM, N_HEADS, FF_DIM, NUM_ENCODER_LAYERS, num_classes
    ).to(_device)
    m.load_state_dict(state_dict)
    m.eval()
    _model = m
    return _model, _device


# =========================================================
# Helpers
# =========================================================

def word_tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())


def encode_query(query: str, vocab_dict=None):
    import torch
    import category.classifier_routes as _self
    
    v = vocab_dict if vocab_dict is not None else _self.vocab
    tokens  = word_tokenize(query)
    indices = [v.get(t, v.get("<UNK>", 0)) for t in tokens[:MAX_LEN]]
    padded  = indices + [0] * (MAX_LEN - len(indices))
    return torch.tensor([padded], dtype=torch.long)


def predict_category(query: str, model=None, vocab=None, label_encoder=None) -> str:
    """
    Keyword args accepted for warmup-call compatibility with main.py lifespan.
    Module-level globals are used when args are not provided.
    """
    import torch
    import category.classifier_routes as _self

    m, device = _get_model()
    if model is not None:
        m = model

    _le = label_encoder if label_encoder is not None else _self.label_encoder

    m.eval()
    with torch.no_grad():
        tensor          = encode_query(query, vocab_dict=vocab).to(device)
        output          = m(tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    # Cast to pure Python string to prevent FastAPI JSON serialization errors (np.str_)
    return str(_le.inverse_transform([predicted_index])[0])


def _classify_sync(query: str) -> str:
    """Blocking inference — runs in threadpool, never on the event loop."""
    _get_model()  # ensure loaded
    return predict_category(query)


# =========================================================
# Router
# =========================================================

router = APIRouter(prefix="/api", tags=["Query Classification"])


class QueryInput(BaseModel):
    query: str


async def _handle(input_data: QueryInput):
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query field is required")

    if vocab is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail=f"Classifier unavailable — missing: {', '.join(_missing_files)}",
        )

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_classify_sync, input_data.query),
            timeout=30.0,
        )
        return {"category": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Classifier timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
async def classify_query(input_data: QueryInput):
    """Primary endpoint — POST /api/classify"""
    return await _handle(input_data)


@router.post("/query/category")
async def classify_query_category(input_data: QueryInput):
    """Alias — matches Dashboard frontend call to POST /api/query/category"""
    return await _handle(input_data)


@router.get("/")
async def classifier_home():
    return {
        "message":       "Category Classifier API is running!",
        "ready":         vocab is not None and label_encoder is not None,
        "missing_files": _missing_files,
    }