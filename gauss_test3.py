import os

# --- 关键：必须在导入 diffusers 之前设置镜像，否则可能连接超时 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

# --- 1. 模型加载与配置 ---
# 直接使用 Hugging Face 的 ID，代码会自动通过镜像下载
model_id = "runwayml/stable-diffusion-inpainting"

# 设置调度器 (保持与 v1.5 一致的参数)
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(f">>> Loading model: {model_id} ...")
# 加载模型：使用 fp16 精度以节省显存
ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    scheduler=scheduler
).to(device)


def load_image(image_path, resize=False):
    image = Image.open(image_path).convert("RGB")
    if resize:
        image = image.resize((512, 512))
    return np.array(image)


# --- 2. 核心处理类 (针对 9 通道 Inpainting 模型) ---
class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None
        # Inpainting 特有参数：4(latent) + 1(mask) + 4(masked_image) = 9 通道
        self.mask_latent = None
        self.masked_image_latent = None

    def prev_step(self, model_output, timestep, sample):
        # DDIM 反向步
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        # DDIM 正向步 (Inversion)
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred(self, latents, t, is_forward=True):
        # 构造 Inpainting 专用 9 通道输入
        latent_model_input = torch.cat([latents, self.mask_latent, self.masked_image_latent], dim=1)
        latent_model_input = latent_model_input.to(dtype=self.model.unet.dtype)

        uncond_embeddings, cond_embeddings = self.context.chunk(2)

        # UNet 预测噪声
        noise_pred_uncond = self.model.unet(latent_model_input, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latent_model_input, t, cond_embeddings)["sample"]

        # Guidance Scale 设为 1.0 以保证求逆的可逆性
        noise_pred = noise_pred_uncond + 1.0 * (noise_prediction_text - noise_pred_uncond)

        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def image2latent(self, image):
        if type(image) is np.ndarray:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        latents = self.model.vae.encode(image.to(dtype=self.model.vae.dtype))['latent_dist'].mean
        return latents * 0.18215

    @torch.no_grad()
    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents.to(dtype=self.model.vae.dtype))['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        return (image * 255).astype(np.uint8)

    def init_prompt(self, prompt):
        uncond_input = self.model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(device))[0]
        text_input = self.model.tokenizer([prompt], padding="max_length", max_length=77, truncation=True,
                                          return_tensors="pt")
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=self.model.unet.dtype)

    def set_inpaint_params(self, image_latent):
        # 设置“空掩码”：全 0 表示没有任何区域被 mask，即全图参考
        self.mask_latent = torch.zeros((1, 1, 64, 64), device=device, dtype=self.model.unet.dtype)
        self.masked_image_latent = image_latent.to(dtype=self.model.unet.dtype)

    @torch.no_grad()
    def ddim_loop(self, latent, is_forward=True):
        all_latent = [latent]
        for i in tqdm(range(self.num_ddim_steps)):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward)
            all_latent.append(latent)
        return all_latent

    def invert(self, prompt, start_latent, is_forward):
        self.init_prompt(prompt)
        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]

    @property
    def scheduler(self):
        return self.model.scheduler


# --- 3. 主程序逻辑 ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./steganodataset_v1/0b291cfc2c.jpg.jpg')
    parser.add_argument('--private_key', type=str, default='a gold fish in the aquarium looks around')
    parser.add_argument('--public_key', type=str, default='a cat in the aquarium looks around')
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/3')
    parser.add_argument('--num_steps', type=int, default=50)
    args = parser.parse_args()


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ode = ODESolve(ldm_stable, args.num_steps)

    # 加载图像并处理 (强制 512x512 以匹配 SD v1.5)
    image_gt = load_image(args.image_path, resize=True)
    image_gt_latent = ode.image2latent(image_gt).to(dtype=torch.float16)
    cv2.imwrite(f"{args.save_path}/0_gt.png", cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    # --- 阶段 1: 隐藏信息 (Encoding) ---
    print("\n>>> Phase 1: Encoding message into image...")
    ode.set_inpaint_params(image_gt_latent)
    # Forward: Image -> Noise (using private prompt)
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)

    # Backward: Noise -> Stego Image (using public prompt)
    image_hide_latent = ode.invert(args.public_key, latent_noise, is_forward=False)
    image_hide = ode.latent2image(image_hide_latent)
    cv2.imwrite(f"{args.save_path}/1_stego.png", cv2.cvtColor(image_hide, cv2.COLOR_RGB2BGR))

    # --- 阶段 2: 提取信息 (Decoding/Reveal) ---
    print("\n>>> Phase 2: Extracting message from stego image...")
    image_hide_latent_rec = ode.image2latent(image_hide).to(dtype=torch.float16)
    # 这里的关键：Inpainting 条件要切换成当前的载体图
    ode.set_inpaint_params(image_hide_latent_rec)

    # Forward: Stego Image -> Noise (using public prompt)
    latent_noise_rec = ode.invert(args.public_key, image_hide_latent_rec, is_forward=True)

    # Backward: Noise -> Recovered Image (using private prompt)
    image_reverse_latent = ode.invert(args.private_key, latent_noise_rec, is_forward=False)
    image_reverse = ode.latent2image(image_reverse_latent)
    cv2.imwrite(f"{args.save_path}/2_recovered.png", cv2.cvtColor(image_reverse, cv2.COLOR_RGB2BGR))

    print(f"\nDone! Results saved in: {args.save_path}")