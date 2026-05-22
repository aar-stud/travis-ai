import os
import warnings
from contextlib import asynccontextmanager

# Suppress PyTorch nested tensor warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*transformer.*")
warnings.filterwarnings("ignore", message=".*nested tensor.*")

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# Configuration
# =========================================================

PORT = int(os.environ.get("PORT", 5001))
CHROMA_PATH = "./knowledge_base/chroma_store"

# =========================================================
# App
# =========================================================

_failed_services = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup + shutdown lifecycle.
    Handles:
    - auto-ingest if ChromaDB missing
    - model warmup
    - service initialization
    """

    print("\n================================================")
    print(" TRAVIS Multi-Service AI API Starting")
    print("================================================\n")

    # =====================================================
    # Auto-create ChromaDB if missing
    # =====================================================

    try:
        if not os.path.exists(CHROMA_PATH):
            print("[startup] ChromaDB not found.")
            print("[startup] Running RAG ingestion pipeline...\n")

            from rag.ingest import main as ingest_main

            ingest_main()

            print("[startup] ChromaDB successfully created.\n")

        else:
            print("[startup] Existing ChromaDB found.\n")

    except Exception as e:
        print(f"[startup] ChromaDB initialization failed: {e}\n")

    # =====================================================
    # Warmup Models
    # =====================================================

    print("[startup] Warming up ML models...\n")

    # 1. QA Model
    try:
        from bank.qa_routes import generate_response

        generate_response("what is a bank account")

        print("[startup] QA model warm.")
    except Exception as e:
        print(f"[startup] QA warmup failed: {e}")

    # 2. Translation
    try:
        from translation.translate_routes import _load_custom_model

        if _load_custom_model():
            print("[startup] Translation model warm.")
        else:
            print("[startup] Translation fallback enabled.")
    except Exception as e:
        print(f"[startup] Translation warmup failed: {e}")

    # 3. Intent Classifier
    try:
        from category.classifer_routes import (
            predict_category,
            model,
            vocab,
            label_encoder,
        )

        predict_category(
            "what is my account balance",
            model,
            vocab,
            label_encoder,
        )

        print("[startup] Classifier model warm.")
    except Exception as e:
        print(f"[startup] Classifier warmup failed: {e}")

    # 4. SentenceTransformer Embedding Model
    try:
        from rag.embedder import get_model

        get_model()

        print("[startup] RAG embedding model warm.")
    except Exception as e:
        print(f"[startup] Embedder warmup failed: {e}")

    # 5. ChromaDB Connection
    try:
        from rag.retriever import _get_collection

        col = _get_collection()

        print(f"[startup] ChromaDB connected — {col.count()} chunks indexed.")
    except Exception as e:
        print(f"[startup] ChromaDB connection failed: {e}")

    # 6. RAG Pipeline Dry Run
    try:
        from rag.embedder import embed_query
        from rag.retriever import retrieve

        vec = embed_query("hello")
        retrieve(vec, top_k=1)

        print("[startup] RAG pipeline warm.")
    except Exception as e:
        print(f"[startup] RAG warmup failed: {e}")

    print("\n[startup] TRAVIS AI API Ready.\n")

    yield

    # =====================================================
    # Shutdown
    # =====================================================

    print("\n[shutdown] TRAVIS AI API shutting down...\n")


app = FastAPI(
    title="TRAVIS Multi-Service AI API",
    version="2.1.0",
    lifespan=lifespan,
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Safe Router Loading
# =========================================================


def _try_include(label, import_fn):
    try:
        router = import_fn()
        app.include_router(router)

        print(f"[router] OK   — {label}")

    except Exception as e:
        _failed_services.append(label)

        print(f"[router] SKIP — {label}: {e}")


# =========================================================
# Routers
# =========================================================

# 1. QA
def _load_qa():
    from bank.qa_routes import qa_router

    return qa_router


_try_include("qa (seq2seq)", _load_qa)

# 2. Translation
def _load_translation():
    from translation.translate_routes import translation_router

    return translation_router


_try_include("translation (en→te)", _load_translation)

# 3. TTS
def _load_tts():
    from tts.tts_routes import tts_router

    return tts_router


_try_include("tts", _load_tts)

# 4. Classifier
def _load_classifier():
    from category.classifer_routes import router as classifier_router

    return classifier_router


_try_include("classifier", _load_classifier)

# 5. RAG
def _load_rag():
    from rag.rag_routes import rag_router

    return rag_router


_try_include("rag", _load_rag)

# =========================================================
# Routes
# =========================================================


@app.get("/")
async def root():
    return {
        "message": "TRAVIS Multi-Service AI API v2.1",
        "status": "running",
        "failed_services": _failed_services,
        "services": {
            "qa": "/api/predict",
            "classifier": "/api/classify",
            "translation": "/api/translate",
            "tts": "/api/tts",
            "rag": "/api/rag",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if not _failed_services else "degraded",
        "failed_services": _failed_services,
        "port": PORT,
    }


# =========================================================
# Main
# =========================================================

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=PORT,
#         reload=False,
#     )