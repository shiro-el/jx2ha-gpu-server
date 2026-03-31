# Testing Patterns

**Analysis Date:** 2026-03-31

## Test Framework

**Not Detected**

The codebase does not include any formal testing framework (pytest, unittest, etc.) or test files. Scripts are designed as remote execution tasks through ClearML rather than standalone modules that would benefit from unit testing.

**Testing Approach:**
- Scripts are validated through execution on ClearML's remote GPU queue (`junha-5090`)
- Output validation happens through ClearML artifact inspection
- No automated test suite or CI/CD testing pipeline detected

## ClearML Integration Testing Pattern

**Validation Method:**
- Each script is a self-contained task that uploads artifacts for inspection
- Results logged to ClearML dashboard for review:
  - JSON output files: `task.upload_artifact("parsed_json", artifact_object=...)`
  - Markdown output files: `task.upload_artifact("parsed_markdown", artifact_object=...)`
  - Image artifacts: `task.upload_artifact("portrait_1", artifact_object=...)`

**Progress Monitoring:**
```python
# Print statements provide console-level progress tracking
print(f"Generating image {i+1}: {prompt[:50]}...")
print(f"페이지 {i + 1}/{len(images)} 파싱 중...")
print(f"  → {block_count}개 블록 추출")
```

## Data Validation Patterns

**Type Checking:**
Scripts include defensive type checks within processing loops:

```python
# From mineru_parse.py and mineru_parse_ir.py
block_count = len(blocks) if isinstance(blocks, list) else "?"
if isinstance(blocks, list):
    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type", "text")
            content = block.get("content", str(block))
```

**Defensive Dict Access:**
Use `.get()` with fallback values to handle missing keys:

```python
block_type = block.get("type", "text")
content = block.get("content", str(block))
blocks = page.get("blocks", [])
```

## Output Validation

**File Existence Checks:**
```python
# From mineru_parse.py
sample_path = Path("sample.pdf")
if not sample_path.exists():
    print(f"샘플 PDF 다운로드 중: {sample_url}")
    urllib.request.urlretrieve(sample_url, str(sample_path))
```

**Output Directory Creation:**
```python
output_path = Path("output")
output_path.mkdir(exist_ok=True)
```

## Manual Testing Steps (Inferred from Scripts)

**FLUX Portrait Generation (flux2_portrait.py):**
1. Initialize ClearML Task with project/task names
2. Execute remotely on RTX 5090 queue
3. Load FLUX.2-Klein-4B model from HuggingFace
4. Generate 3 portrait images with different prompts
5. Verify images saved locally
6. Upload artifacts to ClearML (portrait_1, portrait_2, portrait_3)
7. Manual inspection of generated images in ClearML dashboard

**MinerU Document Parsing (mineru_parse.py):**
1. Initialize ClearML Task with project/task names
2. Execute remotely on RTX 5090 queue
3. Download sample PDF (MinerU2.5 paper)
4. Convert PDF to images
5. Parse each page with MinerUClient
6. Extract blocks and validate output structure
7. Save JSON and markdown representations
8. Upload artifacts to ClearML (parsed_json, parsed_markdown)
9. Manual inspection of parsed content structure

**IR Document Parsing (mineru_parse_ir.py):**
1. Upload local PDF to ClearML artifacts
2. Initialize ClearML Task with project/task names
3. Execute remotely on RTX 5090 queue
4. Retrieve uploaded PDF from artifacts
5. Load MinerU model with vLLM backend
6. Convert PDF to images
7. Parse each page with MinerUClient
8. Extract blocks and structure
9. Save JSON and markdown representations
10. Upload artifacts to ClearML (parsed_json, parsed_markdown)

## Artifact-Based Testing

**Output Files:**
All scripts validate success by producing artifacts:

**JSON Output:**
```python
json_path = output_path / "parsed.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
task.upload_artifact("parsed_json", artifact_object=str(json_path))
```

**Markdown Output:**
```python
md_path = output_path / "parsed.md"
with open(md_path, "w", encoding="utf-8") as f:
    for page in results:
        f.write(f"# Page {page['page']}\n\n")
        # Block processing and writing
task.upload_artifact("parsed_markdown", artifact_object=str(md_path))
```

## Dependencies for Testing

**No Test Dependencies Configured**

The scripts use only runtime dependencies:
- `torch` - Deep learning framework
- `clearml` - Task orchestration and artifact management
- `diffusers` - Flux model pipeline
- `transformers` - Model loading (AutoProcessor, AutoModel)
- `mineru_vl_utils` - Document parsing client
- `PyMuPDF` - PDF processing
- `PIL` - Image handling
- `vllm` - Language model serving (optional, for IR parsing)

**No dev/test dependencies detected:**
- No pytest installation
- No mock library imports
- No test fixture frameworks
- No coverage tools configured

## Reproducibility & Seeding

**Random Seed:**
FLUX script uses explicit seeding for reproducibility:

```python
generator=torch.Generator(device=device).manual_seed(42 + i)
```

This ensures portrait generation produces consistent results across runs.

## ClearML Queue Testing

**Remote Execution Validation:**
- Queue name: `"junha-5090"` (RTX 5090 GPU queue)
- Execution triggered by: `task.execute_remotely(queue_name="junha-5090")`
- If queue unavailable, scripts will fail at this point with ClearML error
- Success validated by appearance of artifacts in ClearML dashboard

## Missing Test Coverage

**Areas Without Formal Testing:**
- Model loading error handling
- PDF download failures
- GPU out-of-memory conditions
- Malformed input handling
- ClearML connectivity issues
- Network interruptions during long-running tasks

**Risk Level:** High - Production scripts lack error recovery patterns

---

*Testing analysis: 2026-03-31*
