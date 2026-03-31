# Codebase Concerns

**Analysis Date:** 2026-03-31

## Tech Debt

**Lack of Error Handling:**
- Issue: No try-catch blocks, exception handling, or error recovery mechanisms across all scripts
- Files: `mineru_parse.py`, `mineru_parse_ir.py`, `flux2_portrait.py`
- Impact: Any network failure (PDF download), model loading error, or image processing failure will crash the entire task without cleanup
- Fix approach: Add try-except blocks around model loading, file I/O, remote API calls, and PDF processing; implement proper logging for errors

**No Input Validation:**
- Issue: Scripts assume successful downloads, model loading, and file operations without validation
- Files: `mineru_parse.py` (lines 43-50), `mineru_parse_ir.py` (lines 23-25)
- Impact: Silent failures or corrupted data if downloads fail partially or files are invalid
- Fix approach: Add validation checks after downloads, file reads, and API calls; verify file integrity before processing

**Hardcoded Values and Credentials:**
- Issue: Queue names, project names, and file paths are hardcoded directly in code
- Files: `mineru_parse.py` (line 10: "junha-5090"), `mineru_parse_ir.py` (line 7: local file path, line 13: "junha-5090"), `flux2_portrait.py` (line 5: "junha-5090")
- Impact: Code is not portable, requires editing source to run on different systems or queues
- Fix approach: Move configuration to environment variables or config files; use `os.getenv()` with defaults

**Resource Cleanup Issues:**
- Issue: Files and connections not properly closed in error scenarios; no context managers used consistently
- Files: `mineru_parse.py` (lines 56-62: fitz.open/close), `mineru_parse_ir.py` (lines 43-49: fitz.open/close)
- Impact: File handles may remain open if exceptions occur during processing, causing resource leaks
- Fix approach: Use `with` statements for all file operations; ensure model cleanup even on failure

**Missing Logging:**
- Issue: Only basic print() statements for debugging; no structured logging framework
- Files: All three scripts use print() exclusively
- Impact: Difficult to debug remote execution on ClearML; no persistent logs for troubleshooting
- Fix approach: Implement Python logging module; configure appropriate log levels for different operations

## Known Bugs

**Unclear Block Type Handling:**
- Symptoms: Block type checking uses string comparisons ("table", "formula") with fallback to str(block) which may not preserve structure
- Files: `mineru_parse.py` (lines 90-103), `mineru_parse_ir.py` (lines 76-90)
- Trigger: When MinerU client returns blocks with unexpected structure or missing "type" field
- Workaround: Code falls back to converting block to string representation, losing semantic information

**Subprocess Call Without Version Pinning:**
- Symptoms: `flux2_portrait.py` installs diffusers from git HEAD without specifying a commit
- Files: `flux2_portrait.py` (line 10)
- Trigger: Breaking changes in diffusers repository after script is written
- Workaround: None - will fail silently if git installation changes

## Security Considerations

**Hardcoded Local File Paths:**
- Risk: Direct filesystem path exposure in code could lead to unintended data access
- Files: `mineru_parse_ir.py` (line 7)
- Current mitigation: None
- Recommendations: Use environment variables for file paths; validate all paths are within expected directories

**Unverified External Downloads:**
- Risk: Arbitrary PDF downloads from URLs without content validation or timeout limits
- Files: `mineru_parse.py` (lines 43-50)
- Current mitigation: None
- Recommendations: Add URL whitelisting, timeout limits, and file size checks before downloading

**No Authentication/Authorization Checks:**
- Risk: ClearML task credentials and queue access not explicitly validated
- Files: `mineru_parse.py` (line 10), `mineru_parse_ir.py` (line 13), `flux2_portrait.py` (line 5)
- Current mitigation: Relies on ClearML environment setup
- Recommendations: Add validation that task initialization succeeds; log queue selection for audit purposes

## Performance Bottlenecks

**Synchronous Model Loading:**
- Problem: Models load sequentially, blocking execution; no async model preparation
- Files: `mineru_parse.py` (lines 20-33), `mineru_parse_ir.py` (lines 29-36)
- Cause: Direct synchronous calls to model loading; no pre-warming or parallel initialization
- Improvement path: Pre-load models during task initialization; implement model caching across runs

**Fixed DPI for PDF Rendering:**
- Problem: All PDFs converted at hardcoded 200 DPI regardless of document type
- Files: `mineru_parse.py` (line 59), `mineru_parse_ir.py` (line 46)
- Cause: One-size-fits-all approach doesn't account for varying document quality requirements
- Improvement path: Implement adaptive DPI selection based on document characteristics or user parameters

**No Batch Processing Optimization:**
- Problem: Images processed one-at-a-time in sequential loop with print output per page
- Files: `mineru_parse.py` (lines 67-72), `mineru_parse_ir.py` (lines 54-59)
- Cause: Simple loop iteration without batching or GPU utilization optimization
- Improvement path: Group images into batches for more efficient model execution; consider parallel page processing

**Single Image Processing Path:**
- Problem: `flux2_portrait.py` generates images sequentially without any parallelization
- Files: `flux2_portrait.py` (lines 29-43)
- Cause: One prompt → one image → save → repeat pattern
- Improvement path: Use async I/O for saving; consider distributed generation if multiple GPUs available

## Fragile Areas

**Model Version Dependencies:**
- Files: `mineru_parse.py`, `mineru_parse_ir.py` (both hardcode "opendatalab/MinerU2.5-2509-1.2B")
- Why fragile: Model weights may be deprecated or removed from Hugging Face; no fallback versions specified
- Safe modification: Pin exact model revision with commit hash; document tested versions; add version negotiation logic
- Test coverage: No validation that model load succeeds before processing starts

**ClearML Task Initialization:**
- Files: All three scripts (Task.init calls)
- Why fragile: Expects ClearML environment to be pre-configured; no validation that initialization succeeds
- Safe modification: Add try-catch around Task.init(); validate task ID before proceeding; check queue exists before execute_remotely()
- Test coverage: No unit tests for ClearML integration; only testable in production ClearML environment

**Document Format Assumptions:**
- Files: `mineru_parse.py`, `mineru_parse_ir.py` (assume valid PDF structure and extractable images)
- Why fragile: Corrupted PDFs, scanned documents, or edge cases cause silent failures in conversion step
- Safe modification: Add PDF validation; implement fallback extraction methods; handle PIL/fitz exceptions
- Test coverage: No test cases for malformed documents or edge cases

**Block Type Handling:**
- Files: `mineru_parse.py` (lines 88-103), `mineru_parse_ir.py` (lines 74-90)
- Why fragile: Assumes specific block structure; falls back to str() conversion which loses data
- Safe modification: Define strict block schema; add validation; implement proper serialization for all block types
- Test coverage: No tests for different block types or missing fields

## Scaling Limits

**Single GPU Bottleneck:**
- Current capacity: 1 GPU queue (junha-5090) - all tasks serialize on same resource
- Limit: Multiple concurrent parsing tasks will queue indefinitely; generation throughput capped by single GPU
- Scaling path: Implement task distribution; add queue auto-scaling; consider multi-GPU node support

**Memory Overhead per Document:**
- Current capacity: Entire document loaded into memory as image list before processing
- Limit: Large PDFs (>500 pages) may exhaust GPU VRAM; no streaming or incremental processing
- Scaling path: Implement streaming PDF processing; process page batches sequentially; add memory monitoring

**Artifact Storage Unbounded:**
- Current capacity: All parsing results and generated images uploaded to ClearML without size limits
- Limit: No cleanup strategy; long-running tasks accumulate artifacts without retention policy
- Scaling path: Implement artifact expiration; add cleanup routines; compress outputs

## Dependencies at Risk

**PyMuPDF (fitz) Dependency:**
- Risk: GPL licensing may conflict with commercial use; library is single-maintainer project
- Impact: If fitz becomes unavailable, PDF conversion has no drop-in replacement
- Migration plan: Add fallback to pypdf2 or pdfplumber; implement abstract PDF converter interface

**Hardcoded transformers/vLLM Backend Selection:**
- Risk: Different backends (transformers vs vLLM) may have different behavior/performance
- Impact: Same code can produce different results depending on backend; migration between backends is manual
- Migration plan: Implement backend abstraction layer; add backend configuration; validate output consistency

**Git-based Installation of Diffusers:**
- Risk: Installing unreleased software from git HEAD introduces undocumented behavior changes
- Impact: flux2_portrait.py may break when upstream changes; no version rollback mechanism
- Migration plan: Pin to specific commit hash; switch to stable PyPI version once available; test before deploying

## Missing Critical Features

**No Result Validation:**
- Problem: Parsed documents accepted without validation; no structural checks on output
- Blocks: Cannot guarantee parsing quality or completeness

**No Retry Logic:**
- Problem: Network failures during downloads, model loading, or API calls cause immediate failure
- Blocks: Production reliability is not achievable; manual intervention required on transient errors

**No Progress Persistence:**
- Problem: If multi-page document processing is interrupted, entire work is lost; no checkpoint system
- Blocks: Cannot safely process very large documents; recovery requires restarting from page 1

**No Output Format Flexibility:**
- Problem: Fixed output formats (JSON + Markdown); no configuration for additional formats or custom extraction
- Blocks: Users cannot get results in required format without post-processing

## Test Coverage Gaps

**No Unit Tests:**
- What's not tested: Model loading, PDF conversion, block extraction, artifact upload
- Files: All three scripts (`mineru_parse.py`, `mineru_parse_ir.py`, `flux2_portrait.py`)
- Risk: Breaking changes in dependencies go undetected; refactoring is high-risk
- Priority: High

**No Integration Tests:**
- What's not tested: ClearML task execution, remote queue scheduling, artifact handling
- Files: All three scripts
- Risk: Code works locally but fails on remote execution; configuration issues not caught
- Priority: High

**No Edge Case Tests:**
- What's not tested: Corrupted PDFs, missing model files, network timeouts, empty documents
- Files: All three scripts
- Risk: Production failures from unusual but realistic input conditions
- Priority: Medium

**No Performance Tests:**
- What's not tested: Throughput with various document sizes, memory usage, GPU utilization
- Files: `mineru_parse.py`, `mineru_parse_ir.py`
- Risk: Unexpected degradation when document scale changes; inability to predict resource requirements
- Priority: Medium

---

*Concerns audit: 2026-03-31*
