# External Integrations

**Analysis Date:** 2026-03-31

## APIs & External Services

**Model Hosting:**
- Hugging Face Hub - Host for pre-trained model weights
  - SDK/Client: `transformers`, `diffusers` libraries with automatic model download
  - Models:
    - `opendatalab/MinerU2.5-2509-1.2B` (Qwen2VL document parsing)
    - `black-forest-labs/FLUX.2-klein-4B` (image generation)
  - Auth: HF token via `HF_TOKEN` env var (implicit in transformers/diffusers)

**arXiv/PDF Sources:**
- Direct HTTP downloads of PDFs
  - `urllib.request` for direct URL fetching in `mineru_parse.py`
  - Example: `https://arxiv.org/pdf/2509.22186`

**GitHub Integration:**
- Git source installation for diffusers library
  - Command: `pip install git+https://github.com/huggingface/diffusers.git` in `flux2_portrait.py`
  - Allows installation of latest unreleased features

## Data Storage

**Artifact Storage:**
- ClearML Artifact Management
  - Primary: JSON parsing results (`parsed.json`)
  - Secondary: Markdown formatted output (`parsed.markdown`)
  - Generated images (`portrait_*.png`)
  - Connection: Via ClearML Task API (`task.upload_artifact()`)
  - Auth: ClearML credentials (stored in env, not in repo)

**File Storage:**
- Local filesystem only for intermediate files
  - PDFs downloaded to local `Path("sample.pdf")`
  - Images saved locally before upload to ClearML
  - Output directory: `output/` (created per-task)

**Databases:**
- None detected - Data persisted via ClearML artifacts and local files

**Caching:**
- None detected - Models downloaded fresh from Hugging Face Hub on each execution

## Authentication & Identity

**ClearML Authentication:**
- Auth Provider: ClearML API (self-hosted or cloud-hosted)
- Implementation: Environment variables (not exposed in repo)
  - Credentials: Implicit in ClearML SDK initialization
  - Queue access: `task.execute_remotely(queue_name="junha-5090")`
  - Artifact upload: `task.upload_artifact()` - authenticated via ClearML session

**Hugging Face Authentication:**
- Auth Provider: HF Token (optional for public models)
- Implementation: Implicit in `transformers`/`diffusers` libraries
  - Environment variable: `HF_TOKEN` (optional for gated models)
  - Used by: Model download and loading

## Monitoring & Observability

**Error Tracking:**
- Not detected - No Sentry, DataDog, or similar integration

**Logs:**
- ClearML Task Console: Built-in logging via ClearML Task
  - All `print()` statements captured in task logs
  - Accessible from ClearML dashboard
  - Task examples: "MinerU2.5 문서 파싱", "IR 문서 파싱 (vLLM)", "FLUX.2-Klein-4B 인물 렌더링"

**Performance Metrics:**
- ClearML built-in task metrics
- No explicit performance tracking libraries detected

## CI/CD & Deployment

**Hosting:**
- ClearML Queue-based distributed execution
- Queue: `junha-5090` (RTX 5090 GPU cluster)
- Execution: Remote via `task.execute_remotely()`

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, or similar

**Deployment:**
- Manual task execution via ClearML
- Tasks submitted from local development environment

## Environment Configuration

**Required env vars:**
- `CLEARML_WEB_HOST` - ClearML API server URL (implicit)
- `CLEARML_API_HOST` - ClearML API endpoint (implicit)
- `CLEARML_FILES_HOST` - ClearML files server (implicit)
- `CLEARML_API_ACCESS_KEY` - API credential (security-sensitive, not in repo)
- `CLEARML_API_SECRET_KEY` - API credential (security-sensitive, not in repo)
- `HF_TOKEN` - Hugging Face token (optional, for gated models)

**ClearML Configuration:**
- Project name: `Someple` (hardcoded in all task initializations)
- Task names: Specific per script (see task definitions)
- Queue name: `junha-5090` (hardcoded in all execute_remotely calls)

**Secrets location:**
- ClearML configuration files (typically `~/.clearml/clearml.conf`)
- Environment variables (not exposed in repo)
- `.env` file: Not detected in repository

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook receivers

**Outgoing:**
- ClearML Task lifecycle events (implicitly)
  - Task start/completion notifications
  - Artifact upload callbacks
  - Not explicitly configured in codebase

## GPU & Hardware Integration

**CUDA/GPU:**
- PyTorch with CUDA via `torch`
- Execution device: `device = "cuda"` (hardcoded in `flux2_portrait.py`)
- Device mapping: `device_map="auto"` for model loading
- Data types: `torch.bfloat16` for model inference (memory optimization)

**Hardware Target:**
- RTX 5090 GPU required
- Queue-based dispatch ensures execution on correct hardware

---

*Integration audit: 2026-03-31*
