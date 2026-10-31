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

# 加载模型
model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
try:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_path_diffusion, use_auth_token=MY_TOKEN,
                                                         scheduler=scheduler).to(device)
except Exception as e:
    print(f"模型加载警告: {e}。")
    ldm_stable = None

if ldm_stable:
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass
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


def read_mask(mask_path_or_paths: Union[str, List[str]], dest_size=(64, 64), invert_mask=False, mask_target='white'):
    """支持读取单张或多张Mask并自动融合"""
    paths = mask_path_or_paths if isinstance(mask_path_or_paths, list) else [mask_path_or_paths]

    combined_mask = None
    for path in paths:
        if not os.path.exists(path):
            print(f"警告: 找不到掩码文件 {path}")
            continue
        org_mask = Image.open(path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255.0

        if combined_mask is None:
            combined_mask = mask
        else:
            # 如果白色是背景(主体是黑色)，融合主体需要取最小值(只要有黑就是黑)
            # 如果黑色是背景(主体是白色)，融合主体需要取最大值(只要有白就是白)
            if mask_target == 'white':
                combined_mask = np.minimum(combined_mask, mask)
            else:
                combined_mask = np.maximum(combined_mask, mask)

    if combined_mask is None:
        combined_mask = np.ones((dest_size[1], dest_size[0]))

    combined_mask[combined_mask < 0.5] = 0
    combined_mask[combined_mask >= 0.5] = 1

    if invert_mask:
        combined_mask = 1.0 - combined_mask

    combined_mask = combined_mask[np.newaxis, np.newaxis, ...]
    combined_mask = torch.from_numpy(combined_mask).float().to(device)
    return combined_mask


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
        if self.model:
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

    # 拆分 Mask 路径参数，并使用 nargs='+' 支持传入多个图片路径
    parser.add_argument('--mask_path_step5', nargs='+',
                        default=["/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/1/mask_002.png"],
                        help='步骤5用于物理替换的早期轮廓Mask(可多张)')
    parser.add_argument('--mask_path_step6', nargs='+', default=[
        "/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/3/mask_003.png",
        "/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/3/mask_024.png"
    ], help='步骤7用于最后二次保障清洗的最终轮廓Mask(可多张)')

    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/2')

    # 物理背景图像路径
    parser.add_argument('--target_bg_path', type=str, default='/opt/data/bing/bldm/utils/experiment/image/17/1_gt.png')
    parser.add_argument('--target_bg_path7', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/1/hide.png')

    # Mask控制
    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'],
                        help='控制替换为新背景的区域 (白色为背景)')
    parser.add_argument('--restore_mask_target', type=str, default='white', help='控制还原时变化的区域')
    parser.add_argument('--blur_kernel', type=int, default=15, help='高斯模糊的核大小，控制边缘羽化程度')

    # Prompt设置
    parser.add_argument('--private_key', type=str, default='a cat')
    parser.add_argument('--public_key', type=str, default='a tiger')
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
    # 3. 生成 Hidden Image
    # ==========================
    print("正在生成 Hidden Image (全局变换模式)...")
    ode.lamda_text, ode.lamda_wm = 1.0, 0.0

    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True, mask_latent=None)
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False, mask_latent=None)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))
    image_hidden_latents_initial = image_hidden_latent.clone()

    # 4. 添加 StegaStamp 水印
    print("正在添加 StegaStamp 水印...")
    watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))
    hide_image_encoded_path = os.path.join(args.save_path, "02_hidden_image_encoded.png")

    # ==========================
    # 5. 物理背景替换 (支持多Mask融合)
    # ==========================
    print(f"正在物理替换背景... 使用Mask: {args.mask_path_step5}")
    image_hide_wm_np, _, _ = load_image(hide_image_encoded_path, resize=True)
    image_target_bg_np, _, _ = load_image(args.target_bg_path, resize=True)
    image_target_bg_np7, _, _ = load_image(args.target_bg_path7, resize=True)

    # 读取并融合 Step 5 的多张 Mask
    combined_mask_np_5 = None
    for path in args.mask_path_step5:
        if not os.path.exists(path): continue
        org_mask = Image.open(path).convert("L").resize((512, 512), Image.BILINEAR)
        mask_np = np.array(org_mask).astype(np.float32) / 255.0

        if combined_mask_np_5 is None:
            combined_mask_np_5 = mask_np

    k_size = args.blur_kernel | 1
    mask_blurred_5 = cv2.GaussianBlur(combined_mask_np_5, (k_size, k_size), 0)

    if args.mask_target == 'black':
        replace_mask_5 = mask_blurred_5
    else:
        replace_mask_5 = 1.0 - mask_blurred_5
    replace_mask_5 = np.expand_dims(replace_mask_5, axis=-1)

    bg_replaced_np = (image_target_bg_np * replace_mask_5 + image_hide_wm_np * (1.0 - replace_mask_5))
    bg_replaced_np = np.clip(bg_replaced_np, 0, 255).astype(np.uint8)

    bg_replaced_path = os.path.join(args.save_path, "03_bg_physically_replaced.png")
    cv2.imwrite(bg_replaced_path, cv2.cvtColor(bg_replaced_np, cv2.COLOR_RGB2BGR))

    try:
        acc = evaluate_watermark_accuracy(bg_replaced_path)
        print(f"--> [检测] 物理更换背景后，水印准确率: {acc:.4f}")
    except Exception as e:
        pass

    # ==========================
    # 6. 还原流程与二次清洗保障 (支持多Mask融合)
    # ==========================
    print(f"准备最后还原流程...")
    bg_replaced_latent = ode.image2latent(bg_replaced_np)

    # 潜空间还原时使用的 Latent Mask
    restore_mask_latent = read_mask(args.mask_path_step5, invert_mask=(args.restore_mask_target == 'white'),
                                    mask_target=args.mask_target)

    ode.lamda_text, ode.lamda_wm = 0.5, 0.5
    final_noise = ode.invert(args.reveal_public_key, bg_replaced_latent, is_forward=True,
                             mask_latent=restore_mask_latent,
                             en_latents_reset=image_hidden_latents_initial)
    final_reverse_latent = ode.invert(args.reveal_private_key, final_noise, is_forward=False,
                                      mask_latent=restore_mask_latent)

    # 将潜空间转回图像
    final_image = ode.latent2image(final_reverse_latent)

    raw_save_path = os.path.join(args.save_path, '04_revealed.png')
    cv2.imwrite(raw_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"已保存第6步原始还原图至: {raw_save_path}")

    # ==========================
    # ==========================
    # ==========================
    # 7.【二次保障】执行最终强制对齐清洗 (合并 Step 5 和 Step 6 的所有 Mask)
    # ==========================
    print(f"执行最终强制对齐清洗... 正在合并 Step 5 和 Step 6 的所有 Mask")

    # 初始化一个全黑的底图
    super_combined_mask = np.zeros((512, 512), dtype=np.float32)

    # 1. 合并来自 step5 的所有 Mask
    for path in args.mask_path_step5:
        if not os.path.exists(path): continue
        m = Image.open(path).convert("L").resize((512, 512), Image.BILINEAR)
        m_np = np.array(m).astype(np.float32) / 255.0
        # 只要有一方是白色(1.0)，结果就趋向于白色
        super_combined_mask = np.maximum(super_combined_mask, m_np)

    # 2. 合并来自 step6 的所有 Mask
    for path in args.mask_path_step6:
        if not os.path.exists(path): continue
        m = Image.open(path).convert("L").resize((512, 512), Image.BILINEAR)
        m_np = np.array(m).astype(np.float32) / 255.0
        # 继续取最大值合并
        super_combined_mask = np.maximum(super_combined_mask, m_np)

    # 3. 进行羽化处理
    k_size = args.blur_kernel | 1
    mask_blurred_7 = cv2.GaussianBlur(super_combined_mask, (k_size, k_size), 0)

    # 4. 确定替换区域
    # 根据你的要求：白色是主体（保留），黑色区域进行替换
    # 所以 replace_mask 应该是：1.0 - 白色部分 = 黑色部分
    replace_mask_7 = 1.0 - mask_blurred_7
    replace_mask_7 = np.expand_dims(replace_mask_7, axis=-1)

    # 5. 执行强制替换：背景图 * 1(黑色区域) + 还原图 * 0(白色区域)
    final_image = (image_target_bg_np7 * replace_mask_7 + final_image * (1.0 - replace_mask_7))
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # 保存最终结果
    final_save_path = os.path.join(args.save_path, '05_final_revealed.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    # 最终指标
    psnr_val = calculate_psnr_tensor(image_gt, final_image)
    print(f"流程结束。最终恢复图像 PSNR: {psnr_val:.4f} dB")
    try:
        final_acc = evaluate_watermark_accuracy(final_save_path)
        print(f"最终恢复图的水印提取准确率: {final_acc:.4f}")
    except:
        pass