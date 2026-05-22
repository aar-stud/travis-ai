"""
qa_routes.py - Transformer-based QA for banking domain.

FIXED: All torch imports and model loading are lazy (first request only).
Previously module-level torch.load() corrupted torch._C before main.py's
pre-load guard ran, breaking every other ML service.
"""

import json
import re
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

qa_router = APIRouter(prefix="/api", tags=["Question Answering"])

# vocab is plain JSON — safe to load at import time (no C extensions)
with open(os.path.join(BASE_DIR, "model_artifacts/vocabulary_0.02.json")) as f:
    vocab = json.load(f)

inv_vocab  = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

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
    import numpy as np

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe       = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0)

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :].to(x.device)

    class TransformerQA(nn.Module):
        def __init__(self, vocab_size, d_model=256, nhead=8,
                     num_layers=4, dropout=0.1):
            super().__init__()
            self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoder = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=1024,
                dropout=dropout, batch_first=True,
            )
            dec_layer = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward=1024,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
            self.fc      = nn.Linear(d_model, vocab_size)

        def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None):
            src_emb = self.pos_encoder(self.embedding(src))
            tgt_emb = self.pos_encoder(self.embedding(tgt))
            memory  = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            output  = self.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            return self.fc(output)

    m = TransformerQA(vocab_size).to(_device)
    m.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "model_artifacts/transformer_qa_final_0.02.pth"),
            map_location=_device,
        )
    )
    m.eval()
    _model = m
    return _model, _device


# =========================================================
# Pure-python helpers
# =========================================================

def preprocess(text: str) -> str:
    return re.sub(r"\W", " ", text).lower().strip()


def tokenize(text: str) -> list:
    return [vocab.get(word, vocab[UNK_TOKEN]) for word in text.split()]


def decode(tokens: list) -> str:
    skip = {vocab["<PAD>"], vocab["<EOS>"], vocab["<SOS>"]}
    return " ".join(inv_vocab.get(t, "<UNK>") for t in tokens if t not in skip)


# =========================================================
# Inference
# =========================================================

def generate_response(query: str, max_len: int = 5000) -> str:
    import torch

    model, device = _get_model()

    query        = preprocess(query)
    query_ids    = tokenize(query)
    query_tensor = torch.tensor(query_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask     = (query_tensor == vocab["<PAD>"])
    generated    = [vocab["<SOS>"]]

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask   = torch.triu(
                torch.full((len(generated), len(generated)), float("-inf")),
                diagonal=1,
            ).to(device)
            out        = model(
                query_tensor, tgt_tensor,
                src_key_padding_mask=src_mask,
                tgt_mask=tgt_mask,
            )
            next_token = out[0, -1, :].argmax().item()
            if next_token == vocab["<EOS>"]:
                break
            generated.append(next_token)

    return decode(generated)


# =========================================================
# API
# =========================================================

class QueryRequest(BaseModel):
    query: str


@qa_router.post("/predict")
async def process_query(request_data: QueryRequest):
    query = request_data.query
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Invalid query")
    try:
        response = generate_response(query)
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))