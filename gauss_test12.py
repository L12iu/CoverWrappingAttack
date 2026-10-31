import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Union, List

from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr

# 尝试导入自定义模块
try:
    from encode_image import StegaStampWatermarker
    from decode_image import evaluate_watermark_accuracy
except ImportError:
    print("警告: 未找到相关模块（StegaStamp 或 evaluate），请确保文件在当前路径或 PYTHONPATH 中。")

# ==========================
# 全局配置与模型加载
# ==========================
MY_TOKEN = ''
GUIDANCE_SCALE = 1.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False, steps_offset=1
)

model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
try:
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_path_diffusion, use_auth_token=MY_TOKEN, scheduler=scheduler
    ).to(device)
    if hasattr(ldm_stable, 'disable_xformers_memory_efficient_attention'):
        ldm_stable.disable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"模型加载警告: {e}。")
    ldm_stable = None


# ==========================
# 工具函数
# ==========================
def load_image(image_path, resize=False):
    """加载图片并处理填充或缩放"""
    image = Image.open(image_path).convert("RGB")
    h, w = image.size
    if resize:
        image = image.resize((512, 512))
    else:
        width_padding = -h % 8
        height_padding = -w % 8
        image = ImageOps.expand(image, (0, 0, width_padding, height_padding), fill=None)
    return np.array(image), h, w


def read_mask(mask_paths: Union[str, List[str]], dest_size=(64, 64), invert_mask=False, mask_target='white'):
    """读取并融合多张Mask"""
    paths = mask_paths if isinstance(mask_paths, list) else [mask_paths]
    combined_mask = None

    for path in paths:
        if not os.path.exists(path):
            continue
        mask = Image.open(path).convert("L").resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255.0

        if combined_mask is None:
            combined_mask = mask
        else:
            if mask_target == 'white':
                combined_mask = np.minimum(combined_mask, mask)
            else:
                combined_mask = np.maximum(combined_mask, mask)

    if combined_mask is None:
        combined_mask = np.ones((dest_size[1], dest_size[0]))

    combined_mask = np.where(combined_mask >= 0.5, 1.0, 0.0)
    if invert_mask:
        combined_mask = 1.0 - combined_mask

    combined_mask = torch.from_numpy(combined_mask[np.newaxis, np.newaxis, ...]).float().to(device)
    return combined_mask


def read_physical_mask(mask_paths, dest_size=(512, 512), blur_kernel=15, mask_target='black'):
    """专门为像素级物理替换准备的平滑 Mask"""
    paths = mask_paths if isinstance(mask_paths, list) else [mask_paths]
    combined_mask = None

    for path in paths:
        if not os.path.exists(path): continue
        mask = Image.open(path).convert("L").resize(dest_size, Image.BILINEAR)
        mask_np = np.array(mask).astype(np.float32) / 255.0
        combined_mask = mask_np if combined_mask is None else np.maximum(combined_mask, mask_np)

    if combined_mask is None:
        combined_mask = np.ones(dest_size, dtype=np.float32)

    # 高斯羽化边缘
    k_size = blur_kernel | 1  # 确保是奇数
    blurred_mask = cv2.GaussianBlur(combined_mask, (k_size, k_size), 0)

    # 颜色翻转与维度扩展 (H, W) -> (H, W, 1)
    final_mask = blurred_mask if mask_target == 'black' else 1.0 - blurred_mask
    return np.expand_dims(final_mask, axis=-1)


def calculate_psnr_tensor(image1, image2, max_val=255.0):
    """计算两张图像的PSNR"""

    def to_tensor(img):
        return torch.tensor(img).float().unsqueeze(0).permute(0, 3, 1, 2)

    return psnr(to_tensor(image1), to_tensor(image2), data_range=max_val)


# ==========================
# ODE 求解器
# ==========================
class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        if self.model:
            self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.context = None
        self.context_en = None
        self.lamda_text = 1.0
        self.lamda_wm = 0.5
        self.image_enhide_latents = None

    def step(self, model_output, timestep, sample, is_forward=True):
        step_size = self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        target_t = timestep + step_size if is_forward else timestep - step_size
        target_t = min(max(target_t, -1), 999)

        alpha_prod_t = self.model.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_target = self.model.scheduler.alphas_cumprod[
            target_t] if target_t >= 0 else self.model.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_target) ** 0.5 * model_output

        return alpha_prod_target ** 0.5 * pred_original_sample + pred_sample_direction

    def get_noise_pred(self, latents, t, is_forward=True):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]

        if self.image_enhide_latents is not None:
            if self.context_en is not None:
                _, cond_embeddings_en = self.context_en.chunk(2)
            else:
                cond_embeddings_en = cond_embeddings

            noise_prediction_wm = self.model.unet(self.image_enhide_latents, t, cond_embeddings_en)["sample"]
            noise_pred = noise_pred_uncond + self.lamda_text * (noise_prediction_text - noise_pred_uncond) + \
                         self.lamda_wm * (noise_prediction_wm - noise_pred_uncond)

            noise_en_pred = noise_pred_uncond + 1 * (noise_prediction_wm - noise_pred_uncond)
            self.image_enhide_latents = self.step(noise_en_pred, t, self.image_enhide_latents, is_forward)
        else:
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_prediction_text - noise_pred_uncond)

        return self.step(noise_pred, t, latents, is_forward)

    @torch.no_grad()
    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        return (image.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    @torch.no_grad()
    def image2latent(self, image):
        if isinstance(image, Image.Image): image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4: return image
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        return self.model.vae.encode(image)['latent_dist'].mean * 0.18215

    @torch.no_grad()
    def init_prompt(self, prompt: str, prompt_en: str = None):
        def encode_text(p):
            uncond_input = self.model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            text_input = self.model.tokenizer([p], padding="max_length", max_length=77, truncation=True,
                                              return_tensors="pt")
            return torch.cat([
                self.model.text_encoder(uncond_input.input_ids.to(device))[0],
                self.model.text_encoder(text_input.input_ids.to(device))[0]
            ])

        self.context = encode_text(prompt)
        self.context_en = encode_text(prompt_en) if prompt_en is not None else None

    @torch.no_grad()
    def invert(self, arg1, arg2, arg3=None, is_forward=True, mask_latent=None, en_latents_reset=None):
        if isinstance(arg2, str):
            prompt_en = arg1
            prompt = arg2
            start_latent = arg3
        else:
            prompt_en = None
            prompt = arg1
            start_latent = arg2
            if arg3 is not None:
                is_forward = arg3

        self.init_prompt(prompt, prompt_en)

        if en_latents_reset is not None:
            self.image_enhide_latents = en_latents_reset.clone()
        else:
            self.image_enhide_latents = None  # 确保没有残留

        latent = start_latent.clone().detach()
        source_latent = start_latent.clone() if mask_latent is not None else None

        for i in tqdm(range(self.num_ddim_steps), leave=False):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1] if is_forward else \
                self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward)

            if mask_latent is not None and source_latent is not None:
                latent = latent * mask_latent + source_latent * (1 - mask_latent)

        return latent


# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gt.png')
    parser.add_argument('--stegastamp_model', type=str,
                        default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth')
    parser.add_argument('--mask_path_step5', nargs='+',
                        default=["/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/1/mask_002.png"])
    parser.add_argument('--mask_path_step6', nargs='+',
                        default=["/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/3/mask_003.png",
                                 "/opt/data/jiang/cross/CRoSS/image_jiang/seg_output/3/mask_024.png"])
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/12')
    parser.add_argument('--target_bg_path', type=str, default='/opt/data/bing/bldm/utils/experiment/image/17/1_gt.png')

    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--restore_mask_target', type=str, default='white')
    parser.add_argument('--blur_kernel', type=int, default=15)

    parser.add_argument('--private_key', type=str, default='a cat')
    parser.add_argument('--public_key', type=str, default='a tiger')
    parser.add_argument('--bg_private_key', type=str, default='a tiger')
    parser.add_argument('--bg_public_key', type=str, default='on top of a hill overlooking mountains')
    parser.add_argument('--num_steps', type=int, default=50)

    # 控制像素水印残差叠加的强度系数
    parser.add_argument('--wm_alpha', type=float, default=1.0, help="像素域水印残差的叠加强度")

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # 1. 初始化
    ode = ODESolve(ldm_stable, args.num_steps)
    watermarker = StegaStampWatermarker(args.stegastamp_model, device=device)

    # 2. 读取 GT
    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "01_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    image_gt_latent = ode.image2latent(image_gt)

    # 3. 生成 Hidden Image
    print("3. 正在生成 Hidden Image...")
    ode.lamda_text, ode.lamda_wm = 1.0, 0.0
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

    # 保存初始潜变量供Step 7使用
    image_hidden_latents_initial = image_hidden_latent.clone()

    # 4. 添加 StegaStamp 水印并提取像素残差
    print("4. 正在添加 StegaStamp 水印并提取像素层残差...")
    watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))
    hide_image_encoded_path = os.path.join(args.save_path, "02_hidden_image_encoded.png")

    # 读取打好水印的图片
    image_hide_wm_np, _, _ = load_image(hide_image_encoded_path, resize=True)

    # 计算像素级的水印特征残差
    wm_residual = image_hide_wm_np.astype(np.float32) - image_hidden_np.astype(np.float32)

    # 5. 物理背景替换
    print("5. 正在物理替换背景...")
    image_target_bg_np, _, _ = load_image(args.target_bg_path, resize=True)
    replace_mask_5 = read_physical_mask(args.mask_path_step5, blur_kernel=args.blur_kernel,
                                        mask_target=args.mask_target)

    bg_replaced_np = np.clip((image_target_bg_np * replace_mask_5 + image_hide_wm_np * (1.0 - replace_mask_5)), 0,
                             255).astype(np.uint8)
    bg_replaced_path = os.path.join(args.save_path, "03_bg_physically_replaced.png")
    cv2.imwrite(bg_replaced_path, cv2.cvtColor(bg_replaced_np, cv2.COLOR_RGB2BGR))

    # 6. 生成图4：第一阶段全局还原
    print("6. 全局还原老虎变成山，生成图4...")
    bg_replaced_latent = ode.image2latent(bg_replaced_np)
    ode.lamda_text, ode.lamda_wm = 0.2, 0.8

    noise_step6 = ode.invert(args.bg_private_key, bg_replaced_latent, is_forward=True)
    revealed_step6_latent = ode.invert(args.bg_public_key, noise_step6, is_forward=False)

    image4_np = ode.latent2image(revealed_step6_latent)
    image4_path = os.path.join(args.save_path, '04_revealed_step6.png')
    cv2.imwrite(image4_path, cv2.cvtColor(image4_np, cv2.COLOR_RGB2BGR))

    try:
        acc4 = evaluate_watermark_accuracy(image4_path)
        print(f"--> [检测] 图4的水印提取准确率: {acc4:.4f}")
    except:
        pass

    # 7. 最终还原：基于图4与掩码的二次清洗保障我
    print("7. 基于图4执行最终二次掩码还原...")
    image4_latent = ode.image2latent(image4_np)
    restore_mask_latent = read_mask(args.mask_path_step6, invert_mask=(args.restore_mask_target == 'black'),
                                    mask_target=args.mask_target)

    step7_private_key = "a cat"
    step7_public_key = "a tiger"
    bg_public_key = args.bg_public_key

    # 【还原】保留了原有的双 prompt 前向加噪，传入 en_latents_reset
    noise_step7 = ode.invert(step7_public_key, bg_public_key, image4_latent, is_forward=True,
                             mask_latent=restore_mask_latent,
                             en_latents_reset=image_hidden_latents_initial)

    # 最终反向还原（退回单 prompt 模式照常运行）
    final_reverse_latent = ode.invert(step7_private_key, noise_step7, is_forward=False,
                                      mask_latent=restore_mask_latent)

    final_image = ode.latent2image(final_reverse_latent)

    # 8. 在像素空间将水印残差叠加回去
    print(f"8. 正在叠加像素级水印残差 (alpha={args.wm_alpha})...")
    final_image_with_wm = final_image.astype(np.float32) + args.wm_alpha * wm_residual
    final_image_with_wm = np.clip(final_image_with_wm, 0, 255).astype(np.uint8)

    final_save_path = os.path.join(args.save_path, '05_final_revealed_step7_with_wm.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image_with_wm, cv2.COLOR_RGB2BGR))
    print(f"已保存最终增强水印的还原图至: {final_save_path}")

    # ==========================
    # 最终指标计算
    # ==========================
    psnr_val = calculate_psnr_tensor(image_gt, final_image_with_wm)
    print(f"流程结束。最终恢复图像 PSNR: {psnr_val:.4f} dB")
    try:
        final_acc = evaluate_watermark_accuracy(final_save_path)
        print(f"最终恢复图的水印提取准确率: {final_acc:.4f}")
    except:
        pass