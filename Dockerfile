# -----------------------------------------------------------------------
# CRITICAL: python:3.10-slim — NOT 3.11
# torch==2.1.2 has no prebuilt Linux wheel for Python 3.11.
# -----------------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# libgomp1 — OpenMP threading for scikit-learn, spacy, onnxruntime
# patchelf  — clears PT_GNU_STACK RWE flag on onnxruntime's .so
#             Railway's kernel blocks shared libs with executable stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN addgroup --system travis && adduser --system --ingroup travis travis

COPY requirements.txt .

# Install PyTorch family from the torch CPU index.
# torchtext MUST come from this same index — PyPI has no matching wheel.
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    torchtext==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Smoke-test torch + torchtext before installing anything else
RUN python -c "import torch, torch.nn, torchtext; _ = torch.zeros(1); print('torch:', torch.__version__, '| torchtext:', torchtext.__version__)"

# Install everything else from PyPI (torch family excluded)
RUN grep -v -E "^(torch==|torchvision==|torchaudio==|torchtext==|#|^$)" requirements.txt \
    > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && \
    rm /tmp/req.txt

# Clear executable-stack flag on onnxruntime .so — Railway's kernel
# security policy (seccomp) rejects PT_GNU_STACK RWE shared objects.
RUN patchelf --clear-execstack \
    /usr/local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so

# Pin spacy model — avoids pulling an incompatible latest version
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Full ML stack smoke-test — build fails here if anything is broken
RUN python -c "import torch, torchtext, numpy as np, spacy, transformers, sentence_transformers; import chromadb; _ = chromadb.PersistentClient; print('ML stack OK | torch:', torch.__version__, '| numpy:', np.__version__, '| torchtext:', torchtext.__version__)"

COPY . .

RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/huggingface && \
    chown -R travis:travis /app

RUN find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

USER travis

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# knowledge_base is mounted at runtime:
# docker run -v "<repo>/knowledge_base:/app/knowledge_base" travis-ai

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]