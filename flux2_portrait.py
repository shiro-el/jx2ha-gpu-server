import torch
from clearml import Task

task = Task.init(project_name="Someple", task_name="FLUX.2-Klein-4B 인물 렌더링")
task.execute_remotely(queue_name="junha-5090")

# === 아래부터 RTX 5090에서 실행 ===

import subprocess
subprocess.run(["pip", "install", "git+https://github.com/huggingface/diffusers.git"], check=True)

from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=dtype,
)
pipe.to(device)

prompts = [
    "A portrait of a young Korean woman in soft golden hour lighting, natural skin texture, shallow depth of field, Canon EOS R5, 85mm f/1.4",
    "A close-up portrait of a middle-aged man with weathered face, dramatic side lighting, black and white, studio photography",
    "A full-body portrait of a woman in a red dress walking through a rainy Tokyo street at night, neon reflections, cinematic",
]

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}: {prompt[:50]}...")
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(42 + i),
    ).images[0]

    filename = f"portrait_{i+1}.png"
    image.save(filename)
    task.upload_artifact(f"portrait_{i+1}", artifact_object=filename)
    print(f"Saved: {filename}")

print("Done!")
