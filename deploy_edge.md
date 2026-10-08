# Edge Deployment Guide

This guide covers running DynamicOntology-GraphRAG on edge hardware —
ARM64 single-board computers, fanless mini-PCs, industrial gateways —
with a **local lightweight LLM** doing the inference. Nothing leaves
the device.

---

## 1. Topology

The engine is deliberately split: the container holds the *framework*
(ontology generation, extraction orchestration, verification math,
clustering, beam search), while the **LLM and embedding model run
beside it** on whichever runtime your hardware supports.

```
┌───────────────────────────── edge device ─────────────────────────────┐
│                                                                       │
│  ┌───────────────────┐        HTTP (OpenAI-compatible)               │
│  │  dograph:edge     │ ────────────────────────────────┐             │
│  │  (this container) │                                 │             │
│  │                   │                                 ▼             │
│  │  core/*           │                   ┌─────────────────────────┐ │
│  │  retriever/*      │                   │  local LLM runtime      │ │
│  └───────────────────┘                   │  · Ollama (CPU/GPU)     │ │
│                                          │  · RKLLM (Rockchip NPU) │ │
│                                          └─────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

Two supported local runtimes:

| Runtime | Hardware | Notes |
|---|---|---|
| **Ollama** | any x86_64/ARM64, ≥ 4 GB RAM | Recommended default; `/api/chat` + JSON mode |
| **RKLLM** | Rockchip RK3588/RK3576 NPU | NPU-accelerated; served behind an OpenAI-compatible proxy |

> The framework talks to both through `connectors/llm_backend.py`. Any
> service exposing an OpenAI-compatible `/v1/chat/completions` also
> works via `OpenAIBackend(base_url=...)`.

---

## 2. Build the image

```bash
docker build -t dograph:edge .
```

The image is ~720 MB as built (python:3.11-slim plus the scientific
stack: numpy/scipy/scikit-learn/networkx/pydantic). No compiler
toolchain and no model weights are baked in — weights stay in the LLM
runtime's own storage. If you need it smaller, install from a trimmed
requirements list (drop `scikit-learn`/`leidenalg`/`python-louvain`
when you use the default `GreedyModularityDetector` and
`HashingEmbedder`) — the demo and the full pipeline run without them.

## 3. Run with Ollama on the host

Start Ollama on the device and pull a small instruct model:

```bash
# on the host
ollama serve &                       # default port 11434
ollama pull qwen2.5:7b-instruct      # ~4.7 GB, good JSON discipline
# tighter budget alternative:
# ollama pull llama3.2:3b-instruct   # ~2 GB
```

Then run the container pointing at it:

```bash
docker run --rm \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  dograph:edge
```

On **Linux**, `host.docker.internal` needs `--add-host=host.docker.internal:host-gateway`.

Or as a compose stack (recommended for unattended nodes):

```yaml
# docker-compose.edge.yml
services:
  ollama:
    image: ollama/ollama:latest
    volumes: [ollama:/root/.ollama]
    restart: unless-stopped
  dograph:
    image: dograph:edge
    environment:
      OLLAMA_HOST: http://ollama:11434
    depends_on: [ollama]
    restart: unless-stopped
volumes:
  ollama:
```

```bash
docker compose -f docker-compose.edge.yml up -d
```

### Selecting the model

The backend picks sane defaults, but you can pin explicitly:

```python
from connectors.llm_backend import OllamaBackend, StructuredLLM

llm = StructuredLLM(
    OllamaBackend(
        host="http://ollama:11434",
        default_model="qwen2.5:7b-instruct",
        default_embedding_model="nomic-embed-text",   # ollama pull nomic-embed-text
    )
)
```

> **Structured-output note.** Ollama is used in JSON mode
> (`supports_json_schema() == False`). The framework still validates
> every reply against the Pydantic contract and performs one automatic
> repair round — but small models occasionally fail twice. If you see
> `SchemaValidationError` in logs, prefer a ≥ 7 B instruct model or
> lower `max_classes`/`max_relations` in the ontology config so the
> JSON payload stays small.

## 4. RKLLM on Rockchip NPU (advanced)

For RK3588/RK3576 boards, run the model on the NPU with RKLLM and put
an OpenAI-compatible shim in front of it (e.g. `rkllm-http` style
wrappers), then point the framework at the shim:

```python
from connectors.llm_backend import OpenAIBackend, StructuredLLM

llm = StructuredLLM(
    OpenAIBackend(
        base_url="http://127.0.0.1:8000/v1",   # your RKLLM shim
        api_key="not-needed",
        default_model="qwen2.5-1.5b-rk3588",
    )
)
```

> **Status: documented but not CI-verified.** We do not run Rockchip
> hardware in CI; the integration path above is the same code path as
> any OpenAI-compatible server, but treat it as untested until you've
> run the demo (`python examples/demo.py --live`) on your board. PRs
> with board-specific notes are very welcome.

## 5. Resource sizing

| Profile | RAM | Model | Typical role |
|---|---|---|---|
| Minimal | 2 GB | llama3.2:3b / qwen2.5:1.5b | sensor-gateway summarisation |
| Balanced | 4–8 GB | qwen2.5:7b | site-level ops assistant |
| Comfortable | 16 GB | llama3.1:8b + nomic-embed-text | full GraphRAG offline box |

Framework-side knobs that matter on small devices:

- `configs/default.yaml` → `ontology.max_classes` / `max_relations`
  (smaller JSON replies = fewer failed parses),
- `HierarchicalIndex(hierarchy_levels=2)` on graphs < 500 entities,
- `HashingEmbedder` (default) runs anywhere; swap for
  `nomic-embed-text` via Ollama only when recall matters more than
  footprint.

## 6. Persistence & health

- Snapshots: `EvidenceVerifier.save_graph(..., fmt="pickle")` /
  `GraphPersistence.to_json(...)` — mount a volume at `/app/data` if
  you keep artifacts.
- The container declares a `HEALTHCHECK` that imports the full stack;
  wire your orchestrator to `docker inspect` health status.
- Logs go to stdout/stderr (JSON via `core.utils.logging.configure_logging(json_logs=True)`).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `BackendUnavailable: No LLM backend` | `OLLAMA_HOST` unreachable — check `curl $OLLAMA_HOST/api/tags` from inside the container |
| `SchemaValidationError` repeats | Model too small for the JSON contract — see §3 note |
| `host.docker.internal` unresolved | Linux: add `--add-host=host.docker.internal:host-gateway` |
| Embedding calls fail | You pinned an embedding model that isn't pulled — `ollama pull nomic-embed-text` |
