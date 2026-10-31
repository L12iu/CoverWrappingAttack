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

# 尝试导入自定义模块
try:
    from encode_image import StegaStampWatermarker
    from decode_image import evaluate_watermark_accuracy
    from basicsr.metrics import calculate_psnr, calculate_ssim
except ImportError:
    print("警告: 未找到相关模块（StegaStamp 或 basicsr），请确保文件在当前路径或 PYTHONPATH 中。")

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from torchmetrics.functional import peak_signal_noise_ratio as psnr

# ==========================
# 全局配置
# ==========================
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
MY_TOKEN = ''
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
GUIDANCE_SCALE = 1.0
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 加载模型 (请确保模型路径正确)
model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_path_diffusion, use_auth_token=MY_TOKEN,
                                                     scheduler=scheduler).to(device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


# ==========================
# 工具函数
# ==========================
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
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
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

    def prev_step(self, model_output, timestep, sample):
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
        if context is None: context = self.context
        uncond_embeddings, cond_embeddings = context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]

        if self.image_enhide_latents is not None:
            noise_prediction_wm = self.model.unet(self.image_enhide_latents, t, cond_embeddings)["sample"]
            noise_pred = noise_pred_uncond + self.lamda_text * (
                        noise_prediction_text - noise_pred_uncond) + self.lamda_wm * (
                                     noise_prediction_wm - noise_pred_uncond)
            noise_en_pred = noise_pred_uncond + 1 * (noise_prediction_wm - noise_pred_uncond)
            if is_forward:
                self.image_enhide_latents = self.next_step(noise_en_pred, t, self.image_enhide_latents)
            else:
                self.image_enhide_latents = self.prev_step(noise_en_pred, t, self.image_enhide_latents)
        else:
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_prediction_text - noise_pred_uncond)

        return self.next_step(noise_pred, t, latents) if is_forward else self.prev_step(noise_pred, t, latents)

    @torch.no_grad()
    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        return (image * 255).astype(np.uint8)

    @torch.no_grad()
    def image2latent(self, image):
        if type(image) is Image.Image: image = np.array(image)
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
        uncond_input = self.model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(device))[0]
        text_input = self.model.tokenizer([prompt], padding="max_length", max_length=77, truncation=True,
                                          return_tensors="pt")
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, mask_latent=None, is_forward=True):
        all_latent = [latent]
        source_latent = latent.clone() if mask_latent is not None else None
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps), leave=False):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1] if is_forward else \
            self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            if mask_latent is not None and source_latent is not None:
                latent = latent * mask_latent + source_latent * (1 - mask_latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, prompt, start_latent, is_forward, mask_latent=None, en_latents_reset=None):
        self.init_prompt(prompt)
        if en_latents_reset is not None: self.image_enhide_latents = en_latents_reset.clone()
        latents = self.ddim_loop(start_latent, mask_latent=mask_latent, is_forward=is_forward)
        return latents[-1]


# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gt.png')
    parser.add_argument('--stegastamp_model', type=str,
                        default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth')
    parser.add_argument('--mask_path', type=str, default="/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/1/mask_002.png")
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/5')

    # 物理背景图像路径
    parser.add_argument('--target_bg_path', type=str, default='/opt/data/bing/bldm/utils/experiment/image/17/1_gt.png')

    # Mask控制
    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'],
                        help='控制隐藏时变化的区域')
    parser.add_argument('--restore_mask_target', type=str, default='black', help='控制还原时变化的区域')

    # Prompt设置
    parser.add_argument('--private_key', type=str,
                        default='a cat')
    parser.add_argument('--public_key', type=str,
                        default='a tiger')
    parser.add_argument('--reveal_private_key', type=str,
                        default='a cat sitting on top of a hill overlooking mountains')
    parser.add_argument('--reveal_public_key', type=str,
                        default='a tiger sitting on top of a hill overlooking mountains')

    parser.add_argument('--num_steps', type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.save_path): os.makedirs(args.save_path)

    # 1. 初始化
    ode = ODESolve(ldm_stable, args.num_steps)
    watermarker = StegaStampWatermarker(args.stegastamp_model, device=device)

    # 2. 读取 GT 图像
    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "01_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    image_gt_latent = ode.image2latent(image_gt)

    # ==========================
    # 3. 生成 Hidden Image (改为全局变换，不使用Mask)
    # ==========================
    print("正在生成 Hidden Image (全局变换模式)...")
    ode.lamda_text, ode.lamda_wm = 1.0, 0.0

    # 全局 Forward 到噪声
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True, mask_latent=None)
    # 全局 Backward 到隐藏图
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False, mask_latent=None)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))
    image_hidden_latents_initial = image_hidden_latent.clone()

    # 4. 添加 StegaStamp 水印
    print("正在添加 StegaStamp 水印...")
    watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))
    hide_image_encoded_path = os.path.join(args.save_path, "02_hidden_image_encoded.png")

    # 5. 物理背景替换 (核心修改部分)
    print(f"正在物理替换背景... 目标背景: {args.target_bg_path}")
    image_hide_wm_np, _, _ = load_image(hide_image_encoded_path, resize=True)
    image_target_bg_np, _, _ = load_image(args.target_bg_path, resize=True)

    # 加载 512 尺寸的 Mask 用于像素合成
    # 注意：如果 mask_target 是 black，表示黑色区域是我们要替换的“背景”，白色区域是“物体”
    org_mask = Image.open(args.mask_path).convert("L").resize((512, 512), Image.NEAREST)
    mask_np = np.array(org_mask).astype(np.float32) / 255.0

    # 确定哪些区域换背景：如果 target 是 black，则原图中黑色的地方换成新背景
    if args.mask_target == 'white':
        # mask_np 为 1 的地方换背景，为 0 的地方保留带水印物体
        # 原始 mask 中黑色是 0，我们要让 0 变成 1（即替换区）
        replace_mask = 1.0 - mask_np
    else:
        replace_mask = mask_np

    replace_mask = np.expand_dims(replace_mask, axis=-1)  # [512, 512, 1]

    # 合成公式：结果 = 新背景 * 替换掩码 + 带水印图 * (1 - 替换掩码)
    bg_replaced_np = (image_target_bg_np * replace_mask + image_hide_wm_np * (1.0 - replace_mask)).astype(np.uint8)

    bg_replaced_path = os.path.join(args.save_path, "03_bg_physically_replaced.png")
    cv2.imwrite(bg_replaced_path, cv2.cvtColor(bg_replaced_np, cv2.COLOR_RGB2BGR))

    # 测算物理替换背景后的水印准确率
    try:
        acc = evaluate_watermark_accuracy(bg_replaced_path)
        print(f"--> [检测] 物理更换背景后，水印准确率: {acc:.4f}")
    except Exception as e:
        print(f"--> [错误] 测算准确率失败: {e}")

    # 6. 还原流程
    print(f"准备最后还原流程...")
    bg_replaced_latent = ode.image2latent(bg_replaced_np)
    restore_mask_latent = None
    if os.path.exists(args.mask_path):
        restore_mask_latent = read_mask(args.mask_path, invert_mask=(args.restore_mask_target == 'black'))

    ode.lamda_text, ode.lamda_wm = 0.5, 0.5
    final_noise = ode.invert(args.reveal_public_key, bg_replaced_latent, is_forward=True,
                             mask_latent=restore_mask_latent,
                             en_latents_reset=image_hidden_latents_initial)
    final_reverse_latent = ode.invert(args.reveal_private_key, final_noise, is_forward=False,
                                      mask_latent=restore_mask_latent)
    final_image = ode.latent2image(final_reverse_latent)

    final_save_path = os.path.join(args.save_path, '04_best_revealed.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    # 最终指标
    psnr_val = calculate_psnr_tensor(image_gt, final_image)
    print(f"流程结束。最终恢复图像 PSNR: {psnr_val:.4f} dB")
    try:
        final_acc = evaluate_watermark_accuracy(final_save_path)
        print(f"最终恢复图的水印提取准确率: {final_acc:.4f}")
    except:
        pass