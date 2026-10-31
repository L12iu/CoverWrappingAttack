from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam
from PIL import Image
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
import argparse
import os



scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
GUIDANCE_SCALE = 1.0
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_path_diffusion, use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


def load_image(image_path, left=0, right=0, top=0, bottom=0, resize=False):
    image = Image.open(image_path).convert("RGB")
    h, w = image.size
    if resize:
        image = np.array(image.resize((64, 64)))
    else:
        width_padding = -h % 8
        height_padding = -w % 8

        padded_image = ImageOps.expand(image, (
            0, 0, width_padding, height_padding),
                                       fill=None,
                                       )
        image = np.array(padded_image)
    return image, h, w


class ODESolve:

    def __init__(self, model, NUM_DDIM_STEPS=50):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None

    # 单步扩散计算
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, context)["sample"]
        return noise_pred

    # 带分类器引导的噪声预测
    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        if context is None:
            context = self.context
        guidance_scale = GUIDANCE_SCALE
        uncond_embeddings, cond_embeddings = context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    # 潜在空间与像素空间的转换
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        return text_embeddings

    # @torch.no_grad()
    def ddim_loop(self, latent, is_forward=True):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def save_inter(self, latent, img_name):
        image = self.latent2image(latent)
        cv2.imwrite(img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def invert(self, prompt, start_latent, is_forward):
        self.init_prompt(prompt)
        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]

    def invert_i2n2i(self, prompt1, prompt2, image_start_latent, flip=False):

        self.init_prompt(prompt1)
        latent_i2n = self.ddim_loop(image_start_latent, is_forward=True)
        xT = latent_i2n[-1]

        if flip:
            xT = torch.flip(xT, dims=[2])

        self.init_prompt(prompt2)
        latent_n2i = self.ddim_loop(xT, is_forward=False)

        return self.latent2image(image_start_latent), image_start_latent, self.latent2image(latent_n2i[-1]), latent_n2i[
            -1]


if __name__ == "__main__":
    import torch.nn.functional as F

    # 初始化
    image_hide_path = './output/demo_grad/hide.png'
    image_gt_path = './output/demo_grad/gt.png'
    prompt_private = 'a cat'
    prompt_public = 'a tiger'
    save_path = './output/demo_grad'
    num_steps = 50
    lr = 5e-2
    num_iter = 50

    ode = ODESolve(ldm_stable, num_steps)

    # 加载图像并转换为 latent
    image_hide, _, _ = load_image(image_hide_path, resize=True)
    image_gt, _, _ = load_image(image_gt_path, resize=True)

    image_hide_latent = ode.image2latent(image_hide).detach()
    image_gt_latent = ode.image2latent(image_gt).detach()

    # 同时在像素空间保存原始图像，用于后续的像素级比较
    image_gt_pixel = torch.from_numpy(image_gt).float() / 127.5 - 1
    image_gt_pixel = image_gt_pixel.permute(2, 0, 1).unsqueeze(0).to(device)

    # 初始化 prompt
    ode.init_prompt(prompt_public)
    xT = ode.ddim_loop(image_hide_latent, is_forward=False)[-1]

    # 创建优化变量
    xT_optim = xT.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([xT_optim], lr=lr)

    # 优化 loop
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()

        # 用 prompt_private 生成 reverse latent
        ode.init_prompt(prompt_private)
        latent_pred = ode.ddim_loop(xT_optim, is_forward=False)[-1]

        # 计算潜在空间的损失
        latent_loss = F.mse_loss(latent_pred, image_gt_latent)

        # 将生成的潜在表示转换为像素空间图像
        # 注意：这里需要保留梯度流
        with torch.set_grad_enabled(True):
            # 使用VAE解码器将潜在表示转换为像素空间
            latent_pred_scaled = 1 / 0.18215 * latent_pred
            pixel_pred = ode.model.vae.decode(latent_pred_scaled)['sample']

            # 计算像素空间的损失
            pixel_loss = F.mse_loss(pixel_pred, image_gt_pixel)

            # 总损失可以是潜在空间和像素空间损失的加权和
            # 您可以根据需要调整权重
            loss = 1 * latent_loss + 0 * pixel_loss

        # 反向传播计算梯度
        loss.backward()

        # 更新优化参数
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(
                f"[{i + 1}/{num_iter}] Total Loss: {loss.item():.4f}, Latent Loss: {latent_loss.item():.4f}, Pixel Loss: {pixel_loss.item():.4f}")
            # 保存中间结果
            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    inter_img = ode.latent2image(latent_pred)
                    cv2.imwrite(f"{save_path}/intermediate_{i + 1}.png", cv2.cvtColor(inter_img, cv2.COLOR_RGB2BGR))

    # 在优化完成后，生成最终图像
    with torch.no_grad():
        ode.init_prompt(prompt_private)
        final_latent = ode.ddim_loop(xT_optim, is_forward=False)[-1]
        reverse_img = ode.latent2image(final_latent)
        cv2.imwrite(f"{save_path}/reverse.png", cv2.cvtColor(reverse_img, cv2.COLOR_RGB2BGR))

        # 同时保存原始图像供比较
        gt_img = ode.latent2image(image_gt_latent)
        cv2.imwrite(f"{save_path}/gt.png", cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))