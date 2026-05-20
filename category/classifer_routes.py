from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import pickle
import torch
# from nltk.tokenize import word_tokenize
from typing import Dict
import os
import re

def word_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======= Configuration =======
MAX_LEN = 64
EMBED_DIM = 128
N_HEADS = 4
NUM_ENCODER_LAYERS = 2
FF_DIM = 256
SAVE_PATH = 'best_transformer_model_90.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Load vocab and label encoder =======
with open( os.path.join(BASE_DIR,'vocab_90.pkl'), 'rb') as f:
    vocab = pickle.load(f)

with open( os.path.join(BASE_DIR,'label_encoder_90.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

# ======= Positional Encoding =======
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ======= Transformer Model =======
class TransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ======= Helper Functions =======
def encode_query(query, vocab):
    tokens = word_tokenize(query.lower())
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens[:MAX_LEN]]
    padded = indices + [0] * (MAX_LEN - len(indices))
    return torch.tensor([padded], dtype=torch.long)

def predict_category(query, model, vocab, label_encoder):
    model.eval()
    with torch.no_grad():
        input_tensor = encode_query(query, vocab).to(DEVICE)
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_category = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_category

# ======= Load Model =======
vocab_size = len(vocab)
num_classes = len(label_encoder.classes_)
model = TransformerClassifier(vocab_size, EMBED_DIM, N_HEADS, FF_DIM, NUM_ENCODER_LAYERS, num_classes).to(DEVICE)
model.load_state_dict(torch.load( os.path.join(BASE_DIR, SAVE_PATH), map_location=DEVICE))
model.eval()

# ======= FastAPI Router =======
router = APIRouter(prefix="/api", tags=["Query Classification"])

class QueryInput(BaseModel):
    query: str

@router.post("/classify")
async def classify_query(input_data: QueryInput):
    query = input_data.query
    if not query:
        raise HTTPException(status_code=400, detail="Query field is required")
    try:
        category = predict_category(query, model, vocab, label_encoder)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def classifier_home():
    return {"message": "🔮 Category Classifier API is running!"}
