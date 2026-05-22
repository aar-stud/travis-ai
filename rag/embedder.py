"""
embedder.py — Singleton SentenceTransformer wrapper.

FIXED: sentence_transformers import is now lazy (deferred to first call).
Previously the module-level import triggered torch._C loading before
main.py's pre-load guard ran, corrupting the entire ML stack.

The model is loaded ONCE on first get_model() call and reused for all
subsequent calls — zero cold-start lag on real requests.
Must use the SAME model name as ingest.py.
"""

import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Return the singleton SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        # Lazy import — torch must already be stable before this runs.
        # In production this is guaranteed because main.py's module-level
        # torch pre-load block runs before any router imports get_model().
        from sentence_transformers import SentenceTransformer

        print(f"[embedder] Loading '{EMBED_MODEL}' ...")
        _model = SentenceTransformer(EMBED_MODEL)
        # Warm up internal tokenizer cache with a dummy encode
        _model.encode("warmup", convert_to_numpy=True)
        print("[embedder] Model ready.")
    return _model


def embed_query(text: str) -> list:
    """Embed a single query string. Returns a Python list of floats."""
    vec: np.ndarray = get_model().encode(text, convert_to_numpy=True)
    return vec.tolist()