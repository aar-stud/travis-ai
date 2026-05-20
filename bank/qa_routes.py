import torch
import torch.nn as nn
import json
import re
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Special token constants
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

# Create router for QA service
qa_router = APIRouter(prefix="/api", tags=["Question Answering"])

# Load vocab
with open(os.path.join(BASE_DIR, 'model_artifacts/vocabulary_0.02.json')) as f:
    vocab = json.load(f)

inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text preprocessing
def preprocess(text: str) -> str:
    text = re.sub(r'\W', ' ', text)
    return text.lower().strip()

def tokenize(text: str) -> list:
    return [vocab.get(word, vocab[UNK_TOKEN]) for word in text.split()]

def decode(tokens):
    return ' '.join([
        inv_vocab.get(t, '<UNK>') 
        for t in tokens 
        if t not in [vocab['<PAD>'], vocab['<EOS>'], vocab['<SOS>']]
    ])

def pad_sequence(seq: list, max_len: int, pad_val: int = 0) -> torch.Tensor:
    return torch.tensor(seq + [pad_val] * (max_len - len(seq)), dtype=torch.long).unsqueeze(0).to(device)

# Model Classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerQA(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None):
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        return self.fc(output)

# Load trained model
model = TransformerQA(vocab_size).to(device)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model_artifacts/transformer_qa_final_0.02.pth'), map_location=device))
model.eval()

def generate_response(query, max_len=5000):
    try:
        # print(f"[DEBUG] Raw Query: {query}")
        query = preprocess(query)
        # print(f"[DEBUG] Preprocessed Query: {query}")
        query_ids = tokenize(query)
        # print(f"[DEBUG] Tokenized Query IDs: {query_ids}")
        query_tensor = pad_sequence(query_ids, len(query_ids))
        # print(f"[DEBUG] Padded Query Tensor: {query_tensor}")

        src_mask = (query_tensor == vocab['<PAD>'])

        generated = [vocab['<SOS>']]
        # print(f"[DEBUG] Starting Generation Loop...")

        for step in range(max_len):
            tgt_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask = torch.triu(torch.full((len(generated), len(generated)), float('-inf')), diagonal=1).to(device)

            with torch.no_grad():
                out = model(query_tensor, tgt_tensor, src_key_padding_mask=src_mask, tgt_mask=tgt_mask)
                next_token = out[0, -1, :].argmax().item()

            # print(f"[DEBUG] Step {step}: Next Token = {next_token} ({inv_vocab.get(next_token, '<UNK>')})")

            if next_token == vocab['<EOS>']:
                break
            generated.append(next_token)

        decoded_response = decode(generated)
        print(f"[DEBUG] Final Decoded Response: {decoded_response}")
        return decoded_response

    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in generate_response: {e}")
        print(traceback.format_exc())
        raise

# API request model
class QueryRequest(BaseModel):
    query: str

# API endpoint
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
