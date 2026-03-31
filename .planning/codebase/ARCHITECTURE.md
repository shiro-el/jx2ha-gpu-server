# Architecture

**Analysis Date:** 2026-03-31

## Pattern Overview

**Overall:** Task-Based Remote Execution Pipeline (ClearML)

**Key Characteristics:**
- Script-based task definition with remote GPU execution
- Loose coupling through ClearML task orchestration
- Local task initialization → remote GPU compute → artifact upload pattern
- No API layer or persistent application state
- Stateless ML model execution workflows

## Layers

**Task Definition & Initialization Layer:**
- Purpose: Define computational tasks with metadata and dependencies
- Location: Root level Python scripts (`flux2_portrait.py`, `mineru_parse.py`, `mineru_parse_ir.py`)
- Contains: ClearML Task setup, project/task naming, queue assignment
- Depends on: ClearML SDK
- Used by: ClearML agent daemon (junha-5090 queue)

**Model & Processing Layer:**
- Purpose: Execute ML models and data transformations on GPU
- Location: Lower half of each script (after `task.execute_remotely()`)
- Contains: Model loading, preprocessing, inference, postprocessing
- Depends on: PyTorch, Hugging Face transformers, diffusers, vLLM, MinerU libraries
- Used by: Task definition layer via remote execution

**Artifact Management Layer:**
- Purpose: Store and retrieve intermediate and final outputs
- Location: Artifact upload calls in scripts (`task.upload_artifact()`)
- Contains: JSON results, markdown outputs, image files
- Depends on: ClearML artifact storage backend
- Used by: Task results retrieval and external consumption

## Data Flow

**Document Parsing Flow (mineru_parse.py):**

1. Local task initialization with `Task.init()` in project "Someple"
2. Requirement declaration (`mineru-vl-utils[transformers]`, `PyMuPDF`)
3. Task routes to remote execution on "junha-5090" queue
4. Remote: Load Qwen2VL model and MinerU client
5. Remote: Download sample PDF from URL or accept pre-uploaded artifact
6. Remote: Convert PDF pages to images via PyMuPDF
7. Remote: Extract content blocks page-by-page using MinerU client
8. Remote: Save results as JSON and markdown
9. Remote: Upload artifacts (`parsed_json`, `parsed_markdown`) back to ClearML
10. Results accessible via ClearML dashboard

**IR Document Parsing Flow (mineru_parse_ir.py):**

1. Local task initialization and local PDF artifact upload
2. Requirement declaration (`mineru-vl-utils[vllm]`, `PyMuPDF`)
3. Task routes to remote execution
4. Remote: Retrieve uploaded PDF via `task.artifacts["input_pdf"].get_local_copy()`
5. Remote: Load vLLM-based MinerU client with custom logits processor
6. Remote: Convert PDF to page images
7. Remote: Parse images to blocks using vLLM inference backend
8. Remote: Output to JSON and markdown with ClearML upload

**Image Generation Flow (flux2_portrait.py):**

1. Local task initialization
2. Task routes to remote execution
3. Remote: Install diffusers from Git repository (nightly)
4. Remote: Load FLUX.2-Klein-4B model in bfloat16 precision
5. Remote: Generate portraits from prompts
6. Remote: Save PNG images and upload as artifacts
7. Results accessible in ClearML dashboard

**State Management:**
- No persistent state across executions
- Task metadata (project, task name) stored in ClearML backend
- Artifacts serve as primary state container
- Each script execution is independent and idempotent (with seeded generators)

## Key Abstractions

**ClearML Task:**
- Purpose: Encapsulates a computational unit with metadata and execution context
- Examples: `flux2_portrait.py`, `mineru_parse.py`, `mineru_parse_ir.py`
- Pattern: `Task.init()` → declare requirements → `execute_remotely()` → compute → `upload_artifact()`

**ML Pipeline (Model + Inference):**
- Purpose: Represents a complete model inference workflow
- Examples: Diffusers Flux2Klein pipeline, Qwen2VL with MinerU client, vLLM MinerU engine
- Pattern: Model loading → preprocessing → inference → postprocessing

**Artifact Container:**
- Purpose: Decouple task results from execution environment
- Examples: PNG images, JSON parsing results, markdown outputs
- Pattern: Local file creation → ClearML upload → dashboard retrieval

## Entry Points

**flux2_portrait.py:**
- Location: `/Users/jae/Documents/GitHub/jx2ha-gpu-server/flux2_portrait.py`
- Triggers: Manual execution via `python flux2_portrait.py` or ClearML scheduler
- Responsibilities: Initialize portrait generation task, configure prompts, manage FLUX.2 model pipeline, save and upload PNG artifacts

**mineru_parse.py:**
- Location: `/Users/jae/Documents/GitHub/jx2ha-gpu-server/mineru_parse.py`
- Triggers: Manual execution or ClearML scheduler
- Responsibilities: Initialize document parsing task, load Qwen2VL model, perform two-step extraction, save JSON and markdown outputs

**mineru_parse_ir.py:**
- Location: `/Users/jae/Documents/GitHub/jx2ha-gpu-server/mineru_parse_ir.py`
- Triggers: Manual execution or ClearML scheduler
- Responsibilities: Initialize IR document task, upload local PDF, load vLLM-based parser, extract and save results

## Error Handling

**Strategy:** Subprocess failures and model loading errors surface immediately to task execution

**Patterns:**
- `subprocess.run(..., check=True)` - Halt on pip install failure (flux2_portrait.py line 10)
- Implicit exception propagation - Model loading failures (lines 20-24, mineru_parse.py) abort task
- Silent fallback - Generic block extraction with type checking and fallback string representation (mineru_parse_ir.py lines 76-88)

## Cross-Cutting Concerns

**Logging:** Print statements to stdout, captured by ClearML task logs

**Validation:** Type checking on block structures before markdown generation (isinstance checks for dict vs list)

**Authentication:** ClearML credentials via environment (not explicit in scripts). GPU queue routing via hardcoded "junha-5090" queue name.

**Model Loading:** Explicit device mapping (`device="cuda"`, `device_map="auto"`) for GPU targeting

---

*Architecture analysis: 2026-03-31*
