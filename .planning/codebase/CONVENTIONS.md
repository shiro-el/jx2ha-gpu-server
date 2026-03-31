# Coding Conventions

**Analysis Date:** 2026-03-31

## Naming Patterns

**Files:**
- Python scripts use snake_case with descriptive names: `flux2_portrait.py`, `mineru_parse.py`, `mineru_parse_ir.py`
- Files are named to reflect their primary task (rendering, parsing, parsing with specific backend)

**Variables:**
- Local variables use snake_case: `task`, `sample_url`, `sample_path`, `pdf_path`, `output_path`, `json_path`, `md_path`
- ClearML Task objects use descriptive names: `task` (standard convention for ClearML)
- Device and dtype variables use lowercase: `device = "cuda"`, `dtype = torch.bfloat16`
- Counters and indices use lowercase: `i` for loop iteration, `page_count`, `block_count`

**Functions:**
- No custom functions defined in scripts; however, API calls follow library naming: `Task.init()`, `pipe()`, `client.two_step_extract()`
- Method calls follow library conventions (CamelCase for class names, lowercase for methods)

**Types/Classes:**
- Import statements use class names as provided by libraries: `Flux2KleinPipeline`, `Qwen2VLForConditionalGeneration`, `MinerUClient`
- Enum-like values use lowercase strings: `dtype="auto"`, `backend="transformers"`, `backend="vllm-engine"`

## Code Style

**Formatting:**
- Lines wrap naturally without enforced length limits visible in configuration
- Indentation: 4 spaces (standard Python convention)
- Multiple statements per block shown in sequence without excessive blank lines

**Comments:**
- Korean language comments used alongside English:
  - `# 모델 로딩 중...` (Model loading...)
  - `# === 아래부터 RTX 5090에서 실행 ===` (From below, execute on RTX 5090)
  - `# ClearML agent venv에 패키지 추가` (Add packages to ClearML agent venv)
- Section separators use `# ============================================================`
- Comments are concise and placed above the code they describe

**Print Statements:**
- Used for progress tracking: `print(f"Generating image {i+1}: {prompt[:50]}...")`
- Include descriptive messages with progress indicators: `print("모델 로딩 중...")`, `print("완료!")`
- String interpolation uses f-strings: `f"페이지 {i + 1}/{len(images)} 파싱 중..."`

## Import Organization

**Order:**
1. Standard library imports: `torch`, `subprocess`, `json`, `urllib.request`
2. Third-party framework imports: `diffusers`, `clearml`, `PIL`, `pathlib`
3. Domain-specific imports: `transformers`, `mineru_vl_utils`, `vllm`

**Pattern:**
```python
import torch
from clearml import Task
import subprocess
from diffusers import Flux2KleinPipeline
import json
from pathlib import Path
from PIL import Image
```

**Path Usage:**
- `pathlib.Path` used for file operations: `Path("sample.pdf")`, `output_path = Path("output")`
- Path existence checks: `if not sample_path.exists():`
- Directory creation with exist_ok: `output_path.mkdir(exist_ok=True)`

## Module Design

**ClearML Initialization Pattern:**
- All scripts follow standard pattern:
  1. Initialize Task: `task = Task.init(project_name="...", task_name="...")`
  2. Add requirements: `task.add_requirements("package")`
  3. Execute remotely: `task.execute_remotely(queue_name="junha-5090")`
  4. Run actual code below execution marker

**Error Handling:**
- Subprocess execution: `subprocess.run(["pip", "install", "..."], check=True)`
- File operations: Existence checks before operations: `if not sample_path.exists():`
- Type checking in block processing: `if isinstance(blocks, list):`, `if isinstance(block, dict):`
- Fallback values: `block.get("type", "text")`, `block.get("content", str(block))`

**Logging & Progress:**
- Status messages printed to console for monitoring
- ClearML artifact upload: `task.upload_artifact(name, artifact_object=path)`
- Progress indicators in loops: `print(f"페이지 {i + 1}/{len(images)} 파싱 중...")`

## Function Design

**No Custom Functions:**
- Scripts are task-oriented with linear execution flow
- Each script represents a complete ClearML task that runs remotely

**Chaining API Calls:**
- Method chaining not used; each operation on separate line
- Device operations explicit: `.to(device)` called after pipeline initialization
- Pipeline calls capture results: `image = pipe(...).images[0]`

## File Operations

**Pattern:**
- Use `pathlib.Path` objects for file operations
- Create output directory: `output_path.mkdir(exist_ok=True)`
- Write JSON: `json.dump(results, f, ensure_ascii=False, indent=2)` (preserves Unicode)
- Write text files: `with open(md_path, "w", encoding="utf-8")`
- File encoding explicitly set: `encoding="utf-8"` for text files

## Configuration Pattern

**Task Configuration:**
- Project names: `"Someple"` (consistent across all scripts)
- Queue names: `"junha-5090"` (RTX 5090 queue, consistent)
- Task names: Descriptive in Korean/English: `"FLUX.2-Klein-4B 인물 렌더링"`, `"MinerU2.5 문서 파싱"`

**Model Configuration:**
- Model names stored as strings: `"black-forest-labs/FLUX.2-klein-4B"`, `"opendatalab/MinerU2.5-2509-1.2B"`
- Dtype specifications: `dtype="auto"`, `torch_dtype=dtype`
- Device settings: `device = "cuda"`, `.to(device)`

## Prompt Storage

**Patterns:**
- Prompts stored in lists: `prompts = [...]`
- Loop through prompts: `for i, prompt in enumerate(prompts):`
- Sample data provided in-code: Full prompts for FLUX, URLs for PDF downloads

---

*Convention analysis: 2026-03-31*
