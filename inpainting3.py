from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

# 1. 定义你的本地模型路径
model_path = "/home/jiang/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e"

# 2. 将 model_path 传给 from_pretrained
# 添加 local_files_only=True 可以强制让程序不尝试连接网络检查更新
pipe = AutoPipelineForInpainting.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True, # SDXL 建议显式开启
    local_files_only=True
).to("cuda")

# --- 以下代码保持不变 ---
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,
  strength=0.99,
  generator=generator,
).images[0]

# 保存一下结果看看
image.save("output.png")