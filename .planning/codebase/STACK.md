# Technology Stack

**Analysis Date:** 2026-03-31

## Languages

**Primary:**
- Python 3.x - All application code for GPU tasks and document processing

## Runtime

**Environment:**
- Python runtime (version specified by system/conda environment)

**Package Manager:**
- pip - Package installation and dependency management
- Lockfile: Not detected (requirements.txt or constraints file not present)

## Frameworks

**Core:**
- ClearML - Distributed task execution, experiment tracking, and artifact management for GPU workloads
- PyTorch - Deep learning framework for ML model execution
- Hugging Face Transformers - Pre-trained model loading and inference
- vLLM - High-throughput LLM serving engine for inference optimization
- Diffusers - Diffusion models library for image generation tasks

**Data Processing:**
- PIL (Pillow) - Image processing and manipulation
- PyMuPDF (fitz) - PDF reading and page-to-image conversion
- OpenDataLab MinerU - Document parsing and visual layout analysis

**Build/Dev:**
- pip with subprocess - Package installation during task execution

## Key Dependencies

**Critical:**
- `torch` - PyTorch core library for GPU computation
- `clearml` - Task orchestration, remote execution on GPU queue "junha-5090", artifact storage
- `transformers` - Qwen2VL and other pre-trained models from Hugging Face
- `mineru-vl-utils[transformers]` - Document parsing with transformer backend (`mineru_parse.py`)
- `mineru-vl-utils[vllm]` - Document parsing with vLLM backend for faster inference (`mineru_parse_ir.py`)
- `vllm` - LLM inference engine used in IR document parsing pipeline
- `diffusers` - FLUX.2 image generation model pipeline
- `PyMuPDF` - PDF-to-image conversion (lighter than poppler)
- `Pillow` - Image format handling and processing

**Infrastructure:**
- `git+https://github.com/huggingface/diffusers.git` - Latest diffusers from source (dynamically installed in `flux2_portrait.py`)

## Configuration

**Environment:**
- GPU queue specified: `junha-5090` (RTX 5090 cluster)
- Tasks execute remotely on GPU via ClearML queue system
- Dynamic package installation via `task.add_requirements()`

**Build:**
- No build configuration files (setup.py, pyproject.toml, requirements.txt)
- Dependencies specified inline via ClearML's `add_requirements()` API

## Platform Requirements

**Development:**
- Local Python environment with torch and ClearML SDK
- Ability to execute ClearML tasks and upload artifacts

**Production:**
- RTX 5090 GPU cluster running ClearML agent
- Queue name: `junha-5090`
- Required CUDA-capable hardware
- Environment variables for ClearML authentication (not detected in repo)

**Models:**
- `opendatalab/MinerU2.5-2509-1.2B` - Document parsing model
- `black-forest-labs/FLUX.2-klein-4B` - Image generation model
- Models downloaded from Hugging Face Hub at runtime

---

*Stack analysis: 2026-03-31*
