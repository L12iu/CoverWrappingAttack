from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
import argparse
import os

# 确保当前目录下有这两个库，或者根据实际情况修改导入路径
try:
    from encode_image import StegaStampWatermarker
    from decode_image import evaluate_watermark_accuracy
    from basicsr.metrics import calculate_psnr, calculate_ssim
except ImportError:
    print("警告: 未找到相关模块，请确保文件存在。")

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from torchmetrics.functional import peak_signal_noise_ratio as psnr

# 全局配置
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
MY_TOKEN = ''
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
GUIDANCE_SCALE = 1.0
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 加载模型
model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
# 如果本地没有模型，可以使用 "runwayml/stable-diffusion-v1-5"
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
        padded_image = ImageOps.expand(image, (0, 0, width_padding, height_padding), fill=None)
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

    # 如果需要反转Mask，0变1，1变0
    if invert_mask:
        mask = 1.0 - mask

    mask = mask[np.newaxis, np.newaxis, ...]
    mask = torch.from_numpy(mask).float().to(device)
    return mask


def calculate_psnr_tensor(image1, image2, max_val=255.0):
    if isinstance(image1, np.ndarray):
        image1 = torch.tensor(image1).float().unsqueeze(0).permute(0, 3, 1, 2)
    if isinstance(image2, np.ndarray):
        image2 = torch.tensor(image2).float().unsqueeze(0).permute(0, 3, 1, 2)
    return psnr(image1, image2, data_range=max_val)


class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None
        self.lamda_text = 1.0
        self.lamda_wm = 0.5
        self.image_enhide_latents = None

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

        uncond_embeddings, cond_embeddings = context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]

        if self.image_enhide_latents is not None:
            noise_prediction_wm = self.model.unet(self.image_enhide_latents, t, cond_embeddings)["sample"]
            w_text = self.lamda_text
            w_wm = self.lamda_wm
            noise_pred = noise_pred_uncond + \
                         w_text * (noise_prediction_text - noise_pred_uncond) + \
                         w_wm * (noise_prediction_wm - noise_pred_uncond)
            noise_en_pred = noise_pred_uncond + 1 * (noise_prediction_wm - noise_pred_uncond)
            if is_forward:
                self.image_enhide_latents = self.next_step(noise_en_pred, t, self.image_enhide_latents)
            else:
                self.image_enhide_latents = self.prev_step(noise_en_pred, t, self.image_enhide_latents)
        else:
            w_text = GUIDANCE_SCALE
            noise_pred = noise_pred_uncond + w_text * (noise_prediction_text - noise_pred_uncond)

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
            if type(image) is Image.Image:
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

        for i in tqdm(range(self.num_ddim_steps), leave=False):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)

            # Mask引导的latent混合
            if mask_latent is not None and source_latent is not None:
                latent = latent * mask_latent + source_latent * (1 - mask_latent)

            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, prompt, start_latent, is_forward, mask_latent=None, en_latents_reset=None):
        self.init_prompt(prompt)
        if en_latents_reset is not None:
            self.image_enhide_latents = en_latents_reset.clone()

        latents = self.ddim_loop(start_latent, mask_latent=mask_latent, is_forward=is_forward)
        return latents[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--gt_path', type=str, default='/opt/data/bing/bldm/utils/experiment/image/17/1_gt.png',
                        help='原始图像路径')
    parser.add_argument('--stegastamp_model', type=str,
                        default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth')
    parser.add_argument('--mask_path', type=str, default="/opt/data/bing/bldm/utils/experiment/image/17/mask.png",
                        help='mask图像路径')
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/6',
                        help='保存结果的文件夹')

    # 掩码目标控制 (根据你的逻辑：通常 black 代表主体，white 代表背景)
    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'],
                        help='步骤3(隐藏)与步骤5(还原)的Mask目标，控制主体变化')
    parser.add_argument('--bg_mask_target', type=str, default='white', choices=['white', 'black'],
                        help='步骤6(换背景)的Mask目标，控制背景区域变化')

    # 核心提示词配置
    parser.add_argument('--private_key', type=str,
                        default='a wooden bench sitting on top of a hill overlooking mountains',
                        help='原始状态描述 (主体A + 背景1)')
    parser.add_argument('--public_key', type=str,
                        default='a white chair sitting on top of a hill overlooking mountains',
                        help='伪装状态描述 (主体B + 背景1)')
    parser.add_argument('--final_bg_key', type=str,
                        default='a wooden bench sitting on a hill overlooking a river and water',
                        help='最终目标描述 (主体A + 背景2)')

    parser.add_argument('--num_steps', type=int, default=50, help='DDIM采样步数')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 1. 初始化模型
    print(">>> 正在初始化模型与加载水印插件...")
    ode = ODESolve(ldm_stable, args.num_steps)
    watermarker = StegaStampWatermarker(args.stegastamp_model, device=device)

    # 2. 读取 GT 图像并转换为 Latent
    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "01_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    image_gt_latent = ode.image2latent(image_gt)

    # ==========================================
    # 3. 隐藏阶段 (Hide): Private -> Public
    # ==========================================
    print(f">>> 步骤3: 生成隐藏图 (Mask目标: {args.mask_target})...")
    mask_latent = None
    if os.path.exists(args.mask_path):
        mask_latent = read_mask(args.mask_path, invert_mask=(args.mask_target == 'black'))

    ode.lamda_text = 1.0
    ode.lamda_wm = 0.0
    # Forward: 得到包含特征的噪声
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True, mask_latent=mask_latent)
    # Backward: 生成伪装图
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False, mask_latent=mask_latent)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

    # 保存用于后续还原引导的初始 Latent
    image_hidden_latents_initial = image_hidden_latent.clone()

    # ==========================================
    # 4. 水印阶段 (Watermark): StegaStamp
    # ==========================================
    print(">>> 步骤4: 为隐藏图添加 StegaStamp 水印...")
    watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))
    # 假设插件默认保存为 02_hidden_image_encoded.png
    hide_image_path = os.path.join(args.save_path, "02_hidden_image_encoded.png")

    # ==========================================
    # 5. 还原阶段 (Reveal): Public -> Private (局部还原主体)
    # ==========================================
    print(f">>> 步骤5: 执行局部还原 (使用水印引导，目标回溯到 private_key)...")
    image_hide_np, _, _ = load_image(hide_image_path, resize=True)
    image_hide_latent = ode.image2latent(image_hide_np)

    # 设定还原时的 Mask (通常与隐藏阶段一致)
    restore_mask_latent = None
    if os.path.exists(args.mask_path):
        restore_mask_latent = read_mask(args.mask_path, invert_mask=(args.mask_target == 'black'))

    # 启用引导参数
    ode.lamda_text = torch.tensor(0.8, dtype=torch.float32, device=device)
    ode.lamda_wm = torch.tensor(0.2, dtype=torch.float32, device=device)

    # 还原流程：从伪装描述(public)回到原始描述(private)
    reveal_noise = ode.invert(args.public_key, image_hide_latent, is_forward=True,
                              mask_latent=restore_mask_latent,
                              en_latents_reset=image_hidden_latents_initial)

    restored_latent = ode.invert(args.private_key, reveal_noise, is_forward=False,
                                 mask_latent=restore_mask_latent)

    restored_np = ode.latent2image(restored_latent)
    restored_path = os.path.join(args.save_path, "03_restored_only.png")
    cv2.imwrite(restored_path, cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR))

    # ==========================================
    # 6. 背景变换阶段 (BG Change): Private -> Final_BG
    # ==========================================
    print(f">>> 步骤6: 变换背景 (Mask目标: {args.bg_mask_target})...")

    # 背景掩码 (控制背景部分变化)
    bg_mask_latent = None
    if os.path.exists(args.mask_path):
        bg_mask_latent = read_mask(args.mask_path, invert_mask=(args.bg_mask_target == 'black'))

    # 关闭水印引导，纯文本驱动
    ode.lamda_wm = 0.0
    ode.lamda_text = 1.0

    # 变换流程：基于还原后的主体，将背景从“山”变为“河流”
    bg_noise = ode.invert(args.private_key, restored_latent, is_forward=True, mask_latent=bg_mask_latent)
    final_latent = ode.invert(args.final_bg_key, bg_noise, is_forward=False, mask_latent=bg_mask_latent)

    final_image = ode.latent2image(final_latent)
    final_save_path = os.path.join(args.save_path, '04_final_revealed_and_bg_changed.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    # ==========================================
    # 7. 指标计算
    # ==========================================
    print(">>> 流程结束，正在计算指标...")
    image_gt_tensor = torch.tensor(image_gt).float().unsqueeze(0).permute(0, 3, 1, 2)
    image_final_tensor = torch.tensor(final_image).float().unsqueeze(0).permute(0, 3, 1, 2)

    res_psnr = calculate_psnr_tensor(image_gt_tensor, image_final_tensor)
    print(f"最终恢复图像与GT的 PSNR: {res_psnr:.4f} dB")

    try:
        wm_accuracy = evaluate_watermark_accuracy(final_save_path)
        print(f"最终结果图的水印提取准确率: {wm_accuracy:.4f}")
    except Exception as e:
        print(f"测算准确率时出错: {e}")