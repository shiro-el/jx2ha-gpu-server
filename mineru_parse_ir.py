import torch
from clearml import Task

task = Task.init(project_name="Someple", task_name="IR 문서 파싱 (vLLM)")

# 로컬 PDF를 ClearML에 업로드
task.upload_artifact("input_pdf", artifact_object="/Users/jae/Downloads/2026. 03 Someple Inc. 투자자용 IR (저용량).pdf")

# ClearML agent venv에 패키지 추가
task.add_requirements("mineru-vl-utils[vllm]")
task.add_requirements("PyMuPDF")

task.execute_remotely(queue_name="junha-5090")

# === 아래부터 RTX 5090에서 실행 ===

import json
from pathlib import Path
from PIL import Image
from vllm import LLM
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor

# 업로드한 PDF 가져오기
pdf_path = task.artifacts["input_pdf"].get_local_copy()
print(f"PDF 경로: {pdf_path}")

# 모델 로딩 (vLLM)
print("모델 로딩 중 (vLLM)...")
llm = LLM(
    model="opendatalab/MinerU2.5-2509-1.2B",
    logits_processors=[MinerULogitsProcessor],
)
client = MinerUClient(
    backend="vllm-engine",
    vllm_llm=llm,
)
print("모델 로딩 완료!")

# PDF → 이미지 변환
import fitz

print("PDF → 이미지 변환 중...")
doc = fitz.open(pdf_path)
images = []
for page in doc:
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(img)
doc.close()
print(f"총 {len(images)} 페이지")

# 페이지별 파싱
results = []
for i, image in enumerate(images):
    print(f"페이지 {i + 1}/{len(images)} 파싱 중...")
    blocks = client.two_step_extract(image)
    results.append({"page": i + 1, "blocks": blocks})
    block_count = len(blocks) if isinstance(blocks, list) else "?"
    print(f"  → {block_count}개 블록 추출")

# JSON 결과 저장 & 업로드
output_path = Path("output")
output_path.mkdir(exist_ok=True)

json_path = output_path / "parsed.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
task.upload_artifact("parsed_json", artifact_object=str(json_path))

# 마크다운 결과 저장 & 업로드
md_path = output_path / "parsed.md"
with open(md_path, "w", encoding="utf-8") as f:
    for page in results:
        f.write(f"# Page {page['page']}\n\n")
        blocks = page.get("blocks", [])
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")
                    content = block.get("content", str(block))
                    if block_type == "table":
                        f.write(f"{content}\n\n")
                    elif block_type == "formula":
                        f.write(f"$$\n{content}\n$$\n\n")
                    else:
                        f.write(f"{content}\n\n")
                else:
                    f.write(f"{block}\n\n")
        else:
            f.write(f"{blocks}\n\n")
        f.write("---\n\n")
task.upload_artifact("parsed_markdown", artifact_object=str(md_path))

print(f"\n완료! 총 {len(results)} 페이지 파싱됨")
