# syntax=docker/dockerfile:1
# =============================================================================
# DynamicOntology-GraphRAG — edge image
#
# A lean runtime-only image for edge deployments (ARM64 SBCs, mini-PCs,
# gateways). The LLM itself is expected to run OUTSIDE this container —
# either on the host (Ollama on :11434) or on an NPU runtime (RKLLM).
# See deploy_edge.md for topology options.
#
# Build:    docker build -t dograph:edge .
# Run:      docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 dograph:edge
# (demo):   docker run --rm dograph:edge
# =============================================================================

# --- Stage 1: builder --------------------------------------------------------
# Wheels are compiled here so the runtime layer stays free of build toolchains.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY core ./core
COPY connectors ./connectors
COPY retriever ./retriever

RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && python -m pip install --no-cache-dir --prefix=/install --no-deps .

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.11-slim AS runtime

# OCI image annotations (retrievable via `docker inspect`).
LABEL org.opencontainers.image.title="DynamicOntology-GraphRAG" \
      org.opencontainers.image.description="Dynamic-ontology GraphRAG engine (edge runtime)" \
      org.opencontainers.image.source="https://github.com/leidingfei2/DynamicOntology-GraphRAG" \
      org.opencontainers.image.licenses="MIT"

# - `PYTHONDONTWRITEBYTECODE`: keep the (often read-only) rootfs clean.
# - `PYTHONUNBUFFERED=1`: logs must stream in real time on edge devices.
# - `PIP_NO_CACHE_DIR=1`: any runtime pip install stays minimal.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

# Non-root user; a fixed UID keeps volume permissions predictable.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin dograph

COPY --from=builder /install /usr/local
WORKDIR /app
COPY examples ./examples
RUN chown -R dograph:dograph /app

USER dograph

# Cheap in-process liveness probe: the imports exercise the full stack
# (pydantic contracts + networkx graph layer) without calling an LLM.
HEALTHCHECK --interval=60s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import core.models, core.evidence_verifier, core.hierarchical_index, retriever.pipeline" || exit 1

CMD ["python", "examples/demo.py"]
