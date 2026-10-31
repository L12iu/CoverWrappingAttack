from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
from PIL import Image, ImageOps

# 尝试导入自定义模块
try:
    from encode_image import StegaStampWatermarker
    from decode_image import evaluate_watermark_accuracy
    from basicsr.metrics import calculate_psnr, calculate_ssim
except ImportError:
    print("警告: 未找到相关模块（StegaStamp 或 basicsr），请确保文件在当前路径或 PYTHONPATH 中。")

from torchmetrics.functional import peak_signal_noise_ratio as psnr

# ==========================
# 全局配置
# ==========================
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
MY_TOKEN = ''
GUIDANCE_SCALE = 1.0
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 加载 Inpainting 模型
model_path_diffusion = "/data/opt_data_backup/data/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
try:
    # 替换为 StableDiffusionInpaintPipeline，并启用 fp16 以节省显存
    ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(
        model_path_diffusion,
        use_auth_token=MY_TOKEN,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler
    ).to(device)
    ldm_stable.safety_checker = None  # 关闭安全检查以加速
except Exception as e:
    print(f"模型加载错误: {e}。")
    ldm_stable = None

if ldm_stable:
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass


# ==========================
# 工具函数 (优化后)
# ==========================
def load_image(image_path, resize=False):
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


def get_combined_mask_np(paths: List[str], dest_size=(512, 512)):
    """提取的通用方法：读取多张 Mask 路径并取最大值融合(只要有一方有白色则为白)"""
    super_mask = np.zeros(dest_size, dtype=np.float32)
    for path in paths:
        if not os.path.exists(path):
            continue
        m_np = np.array(Image.open(path).convert("L").resize(dest_size, Image.BILINEAR)).astype(np.float32) / 255.0
        super_mask = np.maximum(super_mask, m_np)
    return super_mask


def read_mask_latent(mask_path_or_paths: Union[str, List[str]], dest_size=(64, 64), invert_mask=False,
                     mask_target='white'):
    """获取适配于潜空间的 Mask (64x64, 1通道)"""
    paths = mask_path_or_paths if isinstance(mask_path_or_paths, list) else [mask_path_or_paths]

    # 潜空间掩码融合逻辑同像素级，只是尺寸缩小到了 64x64
    combined_mask = get_combined_mask_np(paths, dest_size=dest_size)
    if combined_mask.sum() == 0:
        combined_mask = np.ones(dest_size)  # 兜底逻辑

    # 二值化
    combined_mask[combined_mask < 0.5] = 0
    combined_mask[combined_mask >= 0.5] = 1

    if invert_mask:
        combined_mask = 1.0 - combined_mask

    combined_mask = combined_mask[np.newaxis, np.newaxis, ...]

    # 注意与模型数据类型一致(重要：避免 fp16/fp32 冲突)
    dtype = ldm_stable.unet.dtype if ldm_stable else torch.float32
    return torch.from_numpy(combined_mask).to(device, dtype=dtype)


def calculate_psnr_tensor(image1, image2, max_val=255.0):
    if isinstance(image1, np.ndarray):
        image1 = torch.tensor(image1).float().unsqueeze(0).permute(0, 3, 1, 2)
    if isinstance(image2, np.ndarray):
        image2 = torch.tensor(image2).float().unsqueeze(0).permute(0, 3, 1, 2)
    return psnr(image1, image2, data_range=max_val)


# ==========================
# 核心解算器 (已针对 9 通道 Inpainting 架构完全改造)
# ==========================
class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.dtype = self.model.unet.dtype if self.model else torch.float32

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
        return alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        return alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

    def get_noise_pred(self, latents, t, is_forward=True, context=None, mask_latent=None, masked_image_latent=None):
        if context is None: context = self.context
        uncond_embeddings, cond_embeddings = context.chunk(2)

        # 【核心改进】构建 SD1.5 Inpainting UNet 所需的 9 通道输入
        if mask_latent is not None and masked_image_latent is not None:
            # 1 表示需要 Inpaint (生成)，0 表示保持
            latent_model_input = torch.cat([latents, mask_latent, masked_image_latent], dim=1)
        else:
            # 若无掩码输入，则将其当作普通全局图生成 (所有区域 mask = 1)
            dummy_mask = torch.ones_like(latents[:, :1, :, :])
            dummy_masked_image = torch.zeros_like(latents)
            latent_model_input = torch.cat([latents, dummy_mask, dummy_masked_image], dim=1)

        # 推理
        noise_pred_uncond = self.model.unet(latent_model_input, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latent_model_input, t, cond_embeddings)["sample"]

        if self.image_enhide_latents is not None:
            # 水印解码潜变量同样需要通过 9 通道结构输入
            latent_enhide_input = torch.cat([
                self.image_enhide_latents,
                mask_latent if mask_latent is not None else dummy_mask,
                masked_image_latent if masked_image_latent is not None else dummy_masked_image
            ], dim=1)
            noise_prediction_wm = self.model.unet(latent_enhide_input, t, cond_embeddings)["sample"]

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
        image = self.model.vae.decode(latents.to(self.dtype))['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        return (image * 255).astype(np.uint8)

    @torch.no_grad()
    def image2latent(self, image):
        if type(image) is Image.Image: image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image.to(device, dtype=self.dtype)
        else:
            image = torch.from_numpy(image).to(device, dtype=self.dtype) / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0)
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
        self.context = torch.cat([uncond_embeddings, text_embeddings]).to(self.dtype)
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, mask_latent=None, masked_image_latent=None, is_forward=True):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps), leave=False):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1] if is_forward else \
            self.model.scheduler.timesteps[i]

            # 删除了老旧生硬的 `latent = latent * mask + source * (1-mask)` 逻辑！
            # 掩码已经随潜变量一并送入 UNet 由网络自适应完美融合。
            latent = self.get_noise_pred(latent, t, is_forward, self.context, mask_latent, masked_image_latent)

            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, prompt, start_latent, is_forward, mask_latent=None, en_latents_reset=None):
        self.init_prompt(prompt)
        if en_latents_reset is not None:
            self.image_enhide_latents = en_latents_reset.clone()

        masked_image_latent = None
        if mask_latent is not None:
            # Inpainting 需要提取被保留部分的图像潜变量传入 UNet
            masked_image_latent = start_latent * (1.0 - mask_latent)

        latents = self.ddim_loop(start_latent, mask_latent=mask_latent, masked_image_latent=masked_image_latent,
                                 is_forward=is_forward)
        return latents[-1]


# ==========================
# 主程序
# ==========================




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--gt_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gt.png')
    parser.add_argument('--save_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/inpainting/result')
    parser.add_argument('--stegastamp_model', type=str,
                        default='/data/opt_data_backup/data/140000.pth')

    # 遮罩配置（用于第6步局部还原，若全图还原可忽略或传入全白图）
    parser.add_argument('--mask_path_step5', nargs='+',
                        default=["/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/seg_output/1/mask_002.png"])
    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--restore_mask_target', type=str, default='white')

    # 密钥配置
    parser.add_argument('--private_key', type=str, default='a cat')
    parser.add_argument('--public_key', type=str, default='a tiger')
    parser.add_argument('--reveal_private_key', type=str, default='a cat')
    parser.add_argument('--reveal_public_key', type=str, default='a tiger')

    parser.add_argument('--num_steps', type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================
    # 1. 初始化 (Initialization)
    # ==========================
    print("--- 步骤 1: 初始化 ---")
    # 假设 ldm_stable 已定义
    ode = ODESolve(ldm_stable, args.num_steps)
    # 如果不需要提取物理水印，此步可省略
    watermarker = StegaStampWatermarker(args.stegastamp_model, device=device)

    # ==========================
    # 2. 读取图像 (Load Image)
    # ==========================
    print("--- 步骤 2: 读取源图像 ---")
    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "01_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    image_gt_latent = ode.image2latent(image_gt)

    # ==========================
    # 3. 加密生成 Hidden Image (Encryption)
    # ==========================
    print("--- 步骤 3: 正在执行加密 (ODE Inversion) ---")
    ode.lamda_text, ode.lamda_wm = 1.0, 0.0

    # 前向过程：GT -> Noise (使用私钥)
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True, mask_latent=None)
    # 反向过程：Noise -> Hidden Image (使用公钥)
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False, mask_latent=None)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

    # ==========================
    # 6. 还原提取 (Restoration)
    # ==========================
    print("--- 步骤 6: 正在执行还原 (Extraction) ---")

    # 将加密图转回 latent
    hidden_latent_input = ode.image2latent(image_hidden_np)

    # 读取还原遮罩 (若为全图还原，mask_latent 传 None 即可)
    restore_mask_latent = read_mask_latent(
        args.mask_path_step5,
        invert_mask=(args.restore_mask_target == 'black'),
        mask_target=args.mask_target
    )

    ode.lamda_text, ode.lamda_wm = 0.5, 0.5

    # 还原前向：Hidden -> Noise (使用还原公钥)
    final_noise = ode.invert(args.reveal_public_key, hidden_latent_input, is_forward=True,
                             mask_latent=None)

    # 还原反向：Noise -> Recovered (使用还原私钥)
    final_reverse_latent = ode.invert(args.reveal_private_key, final_noise, is_forward=False,
                                      mask_latent=None)

    # # 还原前向：Hidden -> Noise (使用还原公钥)
    # final_noise = ode.invert(args.reveal_public_key, hidden_latent_input, is_forward=True,
    #                          mask_latent=restore_mask_latent)
    #
    # # 还原反向：Noise -> Recovered (使用还原私钥)
    # final_reverse_latent = ode.invert(args.reveal_private_key, final_noise, is_forward=False,
    #                                   mask_latent=restore_mask_latent)

    final_image = ode.latent2image(final_reverse_latent)

    # 保存结果
    revealed_path = os.path.join(args.save_path, '04_revealed_final.png')
    cv2.imwrite(revealed_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    # 质量评估
    psnr_val = calculate_psnr_tensor(image_gt, final_image)
    print(f"处理完成！")
    print(f"还原图像 PSNR: {psnr_val:.4f} dB")
    print(f"保存路径: {args.save_path}")