# Codebase Structure

**Analysis Date:** 2026-03-31

## Directory Layout

```
jx2ha-gpu-server/
├── flux2_portrait.py           # FLUX.2 image generation task
├── mineru_parse.py             # Document parsing with transformers backend
├── mineru_parse_ir.py          # IR document parsing with vLLM backend
├── test.md                     # MinerU2.5 paper (parsed output reference)
├── .git/                       # Git repository metadata
├── .claude/                    # Claude project configuration
├── .planning/                  # GSD planning artifacts directory
│   └── codebase/              # Architecture and codebase documentation
└── [no src/, lib/, tests/ directories]
```

## Directory Purposes

**Root Level:**
- Purpose: All executable task scripts and metadata files
- Contains: Python scripts, markdown documentation
- Key files: `flux2_portrait.py`, `mineru_parse.py`, `mineru_parse_ir.py`

**.planning/codebase/:**
- Purpose: Stores architecture analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (created by GSD mapping process)
- Committed: Yes (part of project repository)

## Key File Locations

**Entry Points:**
- `flux2_portrait.py`: FLUX.2 Klein 4B image generation task (1.2KB, 46 lines)
- `mineru_parse.py`: MinerU2.5 document parsing with Transformers (3.7KB, 109 lines)
- `mineru_parse_ir.py`: MinerU2.5 IR document parsing with vLLM (3.0KB, 95 lines)

**Configuration:**
- ClearML project: "Someple" (hardcoded in each script)
- GPU queue: "junha-5090" (RTX 5090 server, hardcoded)
- Model references: Hugging Face Model Hub paths (in task initialization and model loading sections)

**Core Logic:**
- Document parsing pipeline: `mineru_parse.py` (lines 18-104)
- Image generation pipeline: `flux2_portrait.py` (lines 14-45)
- IR document parsing with vLLM: `mineru_parse_ir.py` (lines 27-92)

**Testing:**
- No test files present in repository
- Manual testing via ClearML task execution and artifact inspection

## Naming Conventions

**Files:**
- Pattern: `{model}_{task}.py` or `{model}_{task}_{backend}.py`
- Examples:
  - `flux2_portrait.py` - Model name + task type
  - `mineru_parse.py` - Model + task
  - `mineru_parse_ir.py` - Model + task + backend identifier
  - `test.md` - Result/reference documentation

**Functions:**
- Pattern: Procedural, no function extraction (monolithic task scripts)
- No explicit utility functions; logic flows linearly within main execution path

**Variables:**
- Pattern: snake_case throughout
- Examples: `task`, `sample_path`, `sample_url`, `json_path`, `md_path`, `llm`, `client`, `results`, `output_path`

**Types:**
- Imported types are Hugging Face/PyTorch model classes and custom client wrappers
- Examples: `Flux2KleinPipeline`, `Qwen2VLForConditionalGeneration`, `LLM`, `MinerUClient`

## Where to Add New Code

**New GPU Task Script:**
- Create at root level: `/Users/jae/Documents/GitHub/jx2ha-gpu-server/{model_name}_{task}.py`
- Follow pattern from `flux2_portrait.py`:
  1. Import torch and Task from clearml
  2. `Task.init(project_name="Someple", task_name="...")`
  3. `task.add_requirements(...)`
  4. `task.execute_remotely(queue_name="junha-5090")`
  5. Load model and execute
  6. Save outputs and `task.upload_artifact(...)`

**New Local Utility:**
- No dedicated utilities directory exists; add as module-level functions in scripts if small
- For larger utilities, consider extracting to `utils/` directory (not currently present)

**Shared Configuration:**
- Hardcoded values (`project_name="Someple"`, `queue_name="junha-5090"`) live in each script
- Consider extracting to `config.py` if reuse grows

**Test Files:**
- Testing framework not currently used
- If adding tests, create alongside task scripts or in dedicated `tests/` directory with `.test.py` suffix

## Special Directories

**.git/:**
- Purpose: Git version control repository
- Generated: Yes
- Committed: N/A (is the version control root)

**.claude/:**
- Purpose: Claude IDE project configuration
- Generated: Yes (by Claude IDE)
- Committed: Yes (contains settings.local.json)

**.planning/:**
- Purpose: GSD (Guided Software Development) planning and codebase analysis artifacts
- Generated: Yes (by GSD mapper and planner)
- Committed: Yes (versioned with codebase for reference)

## Architecture Decisions

**Monolithic Scripts:**
- Each script is completely self-contained with no shared imports between tasks
- Decision: Allows independent execution and scheduling without dependencies

**No Package Structure:**
- All scripts at root level; no `src/` or `lib/` directories
- Decision: Appropriate for independent, short-lived task scripts meant for manual execution

**ClearML-First Design:**
- All computation delegated to remote GPU via ClearML task framework
- Decision: Separates local orchestration from remote execution, enables queue-based scaling

**Artifact-Based Output:**
- Results stored as ClearML artifacts rather than to shared filesystem
- Decision: Enables dashboard tracking and supports distributed execution model

---

*Structure analysis: 2026-03-31*
