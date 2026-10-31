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

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
GUIDANCE_SCALES = [0.8,1.0, 1.5,2.0,3.0, 4.0]  # 你可以根据需要修改这个列表
MAX_NUM_WORDS = 77
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_path_diffusion, use_auth_token=MY_TOKEN,
                                                     scheduler=scheduler).to(device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


def load_image(image_path, left=0, right=0, top=0, bottom=0, resize=False):
    image = Image.open(image_path).convert("RGB")
    h, w = image.size
    if resize:
        image = np.array(image.resize((512, 512)))
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

    @torch.no_grad()
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


# 主程序
def test_guidance_scale(ldm_model, image_path, prompt1, prompt2, save_dir, num_steps=50):
    image_np, _, _ = load_image(image_path, resize=True)
    image_tensor = image_np

    result_images = []

    for scale in GUIDANCE_SCALES:
        global GUIDANCE_SCALE
        GUIDANCE_SCALE = scale
        print(f"当前 GUIDANCE_SCALE = {scale}")

        ode = ODESolve(ldm_model, num_steps)

        # 获取原始图像 latent
        image_latent = ode.image2latent(image_tensor)

        # 正向推理：加噪
        xT = ode.invert(prompt2, image_latent, is_forward=True)

        # 逆向推理：还原
        latent_recon = ode.invert(prompt1, xT, is_forward=False)
        image_recon = ode.latent2image(latent_recon)

        # 保存图像
        save_path = os.path.join(save_dir, f"reversed_guidance_{scale}.png")
        cv2.imwrite(save_path, cv2.cvtColor(image_recon, cv2.COLOR_RGB2BGR))
        print(f"保存图像到: {save_path}")

        result_images.append(image_recon)

    # 拼接图像
    concat_image = np.concatenate(result_images, axis=1)  # 横向拼接

    # 保存拼接后的图像
    concat_save_path = os.path.join(save_dir, "guidance_scale_comparison.png")
    cv2.imwrite(concat_save_path, cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR))
    print(f"拼接后的图像保存到: {concat_save_path}")

def load_image(image_path, resize=False):
    image = Image.open(image_path).convert("RGB")
    if resize:
        image = np.array(image.resize((512, 512)))  # 假设目标尺寸为 512x512
    else:
        image = np.array(image)
    return image, image.shape[0], image.shape[1]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hideimage_path', type=str, default='/opt/data/jiang/stego_jiang/StegaStamp-pytorch/image_jiang/hide_encoded.png', help='隐藏图像路径')
    parser.add_argument('--private_key', type=str, default='a cat', help='private key prompt')
    parser.add_argument('--public_key', type=str, default='a tiger', help='public key prompt')
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/stego_jiang/StegaStamp-pytorch/image_jiang/guidance_test_cat', help='保存路径')
    parser.add_argument('--num_steps', type=int, default=50, help='DDIM采样步数')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 使用之前加载的模型
    test_guidance_scale(ldm_stable, args.hideimage_path, args.private_key, args.public_key, args.save_path, args.num_steps)
