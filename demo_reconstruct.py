from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np

from torchmetrics.functional import peak_signal_noise_ratio as psnr

import shutil
from torch.optim.adam import Adam
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
import argparse
import os

l1_loss = nn.L1Loss(reduction='mean')
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
MY_TOKEN = ''
LOW_RESOURCE = False
GUIDANCE_SCALE = 1.0
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_path_diffusion = "/opt/data/bing/bldm/models/stablediffusion"
# model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
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


def read_mask(mask_path: str, dest_size=(64, 64), invert_mask=False):
    """
    读取并处理mask图像
    1 表示用生成的内容，0 表示保持原图内容
    增加 invert_mask 选项以自由控制变化区域
    """
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    # ✅ 核心修改：如果需要反转Mask，0变1，1变0
    if invert_mask:
        mask = 1.0 - mask

    mask = mask[np.newaxis, np.newaxis, ...]
    mask = torch.from_numpy(mask).float().to(device)
    return mask


def calculate_psnr(image1, image2, max_val=255.0):
    # 如果输入是NumPy数组，需要转换为Tensor
    if isinstance(image1, np.ndarray):
        image1 = torch.tensor(image1).float().unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
    if isinstance(image2, np.ndarray):
        image2 = torch.tensor(image2).float().unsqueeze(0).permute(0, 3, 1, 2)

    # 计算 PSNR
    return psnr(image1, image2, data_range=max_val)


class ODESolve:

    def __init__(self, model, NUM_DDIM_STEPS=50):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False, steps_offset=1)
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
    def ddim_loop(self, latent, mask_latent=None, is_forward=True):
        all_latent = [latent]
        source_latent = latent.clone() if mask_latent is not None else None
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)

            # ✅ 核心：mask引导的latent混合
            if mask_latent is not None and source_latent is not None:
                latent = latent * mask_latent + source_latent * (1 - mask_latent)

            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, prompt, start_latent, mask_latent=None, is_forward=True):
        self.init_prompt(prompt)
        latents = self.ddim_loop(start_latent, mask_latent, is_forward=is_forward)
        return latents[-1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/opt/data/bing/bldm/utils/experiment/image/17/1_gt.png',
                        help='test image path')
    parser.add_argument('--mask_path', type=str, default="/opt/data/bing/bldm/utils/experiment/image/17/mask.png",
                        help='mask image path (optional)')


    parser.add_argument('--mask_target', type=str, default='white', choices=['white', 'black'],
                        help="控制哪个部分发生变化: 'white'(白色部分变化) 或 'black'(黑色部分变化)")

    parser.add_argument('--private_key', type=str,
                        default='a wooden bench sitting on top of a hill overlooking mountains')
    parser.add_argument('--public_key', type=str,
                        default='a wooden bench sitting on a hill overlooking a river and water')
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing',
                        help='save path')




    #第二步还原
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing/1_hide.png',
    #                     help='test image path')
    # parser.add_argument('--mask_path', type=str, default="/opt/data/bing/bldm/utils/experiment/image/17/mask.png",
    #                     help='mask image path (optional)')
    # # ✅ 核心修改：增加命令行参数控制变化区域
    # parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'],
    #                     help="控制哪个部分发生变化: 'white'(白色部分变化) 或 'black'(黑色部分变化)")
    #
    # parser.add_argument('--private_key', type=str,
    #                     default='a wooden bench sitting on a hill overlooking a river and water')
    # parser.add_argument('--public_key', type=str,
    #                     default='a white chair sitting on a hill overlooking a river and water ')
    # parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing/2',
    #                     help='save path')


#第三步还原
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing/2/1_hide.png',
    #                     help='test image path')
    # parser.add_argument('--mask_path', type=str, default="/opt/data/bing/bldm/utils/experiment/image/17/mask.png",
    #                     help='mask image path (optional)')
    # parser.add_argument('--mask_target', type=str, default='white', choices=['white', 'black'],
    #                     help="控制哪个部分发生变化: 'white'(白色部分变化) 或 'black'(黑色部分变化)")
    #
    # parser.add_argument('--private_key', type=str,
    #                     default='a white chair sitting on a hill overlooking a river and water')
    # parser.add_argument('--public_key', type=str,
    #                     default='a white chair sitting on top of a hill overlooking mountains')
    # parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing/3',
    #                     help='save path')

#第四部
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing/3/1_hide.png',
    #                     help='test image path')
    # parser.add_argument('--mask_path', type=str, default="/opt/data/bing/bldm/utils/experiment/image/17/mask.png",
    #                     help='mask image path (optional)')
    # parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'],
    #                     help="控制哪个部分发生变化: 'white'(白色部分变化) 或 'black'(黑色部分变化)")
    #
    # parser.add_argument('--private_key', type=str,
    #                     default='a white chair sitting on top of a hill overlooking mountains')
    # parser.add_argument('--public_key', type=str,
    #                     default='a wooden bench sitting on top of a hill overlooking mountains')
    # parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/bing/4',
    #                     help='save path')


    parser.add_argument('--num_steps', type=int, default=50, help='sampling step of DDIM')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ode = ODESolve(ldm_stable, args.num_steps)

    image_path = args.image_path
    prompt_1 = args.private_key
    prompt_2 = args.public_key

    rev_prompt_1 = prompt_1
    rev_prompt_2 = prompt_2
    need_flip = False

    mask_latent = None
    if args.mask_path and os.path.exists(args.mask_path):
        # ✅ 核心修改：根据命令行参数决定是否反转mask
        invert_mask = (args.mask_target == 'black')
        mask_latent = read_mask(args.mask_path, invert_mask=invert_mask)
        print(f"Mask loaded successfully. Mask target (changing area): {args.mask_target}")

    offsets = (0, 0, 0, 0)
    image_gt, h, w = load_image(image_path, *offsets, resize=True)

    image_gt_latent = ode.image2latent(image_gt)
    cv2.imwrite("{:s}/1_gt.png".format(args.save_path), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    # hide process
    latent_noise = ode.invert(prompt_1, image_gt_latent, mask_latent, is_forward=True)
    image_hide_latent = ode.invert(prompt_2, latent_noise, mask_latent, is_forward=False)

    # save container image
    image_hide = ode.latent2image(image_hide_latent)
    cv2.imwrite("{:s}/1_hide.png".format(args.save_path), cv2.cvtColor(image_hide, cv2.COLOR_RGB2BGR))

    # reveal process
    image_hide_latent_reveal = ode.image2latent(image_hide)
    latent_noise_recover = ode.invert(rev_prompt_2, image_hide_latent_reveal, mask_latent, is_forward=True)

    image_reverse_latent = ode.invert(rev_prompt_1, latent_noise_recover, mask_latent, is_forward=False)
    image_reverse = ode.latent2image(image_reverse_latent)
    cv2.imwrite("{:s}/1_reverse.png".format(args.save_path), cv2.cvtColor(image_reverse, cv2.COLOR_RGB2BGR))
    loss = l1_loss(latent_noise, latent_noise_recover)
    print("loss:", loss)

    image_gt_tensor = torch.tensor(image_gt).float().unsqueeze(0).permute(0, 3, 1, 2)
    image_reverse_tensor = torch.tensor(image_reverse).float().unsqueeze(0).permute(0, 3, 1, 2)

    # 调用 PSNR 函数，计算并输出 PSNR
    psnr_value = calculate_psnr(image_gt_tensor, image_reverse_tensor)
    print(f"PSNR : {psnr_value:.2f} dB")