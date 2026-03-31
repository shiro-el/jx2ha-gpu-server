import torch
from clearml import Task

task = Task.init(project_name="Someple", task_name="MinerU2.5 문서 파싱")

# ClearML agent venv에 패키지 추가
task.add_requirements("mineru-vl-utils[transformers]")
task.add_requirements("PyMuPDF")

task.execute_remotely(queue_name="junha-5090")

import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient

# 모델 로딩
print("모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B",
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B",
    use_fast=True,
)
client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor,
)
print(f"모델 로딩 완료! Device: {next(model.parameters()).device}")

# ============================================================
# 파싱할 파일 설정
# - ClearML Dataset에서 가져오거나, 직접 URL로 다운로드하거나,
#   task.upload_artifact()로 미리 업로드한 파일을 사용
# ============================================================

# 예시: 샘플 PDF 다운로드 (테스트용)
sample_url = "https://arxiv.org/pdf/2509.22186"  # MinerU2.5 논문 자체를 파싱해보기
sample_path = Path("sample.pdf")

if not sample_path.exists():
    import urllib.request
    print(f"샘플 PDF 다운로드 중: {sample_url}")
    urllib.request.urlretrieve(sample_url, str(sample_path))
    print("다운로드 완료!")

# PDF → 이미지 변환 (PyMuPDF, poppler 불필요)
import fitz  # PyMuPDF

print("PDF → 이미지 변환 중...")
doc = fitz.open(str(sample_path))
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

# JSON 결과 저장 & ClearML에 업로드
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

print(f"\n완료! ClearML 대시보드에서 Artifacts 탭을 확인하세요.")
print(f"총 {len(results)} 페이지 파싱됨")
