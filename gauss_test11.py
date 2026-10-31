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

        # 权重控制系数
        self.lamda_wm = 0.5  # 控制主分支文本引导量 (image4_latent 对应的文本提示词项)
        self.lamda_text = 1.0  # 控制第一分支引导量 (en_latents_reset)
        self.lamda_wm2 = 0.5  # 【新增】控制第二分支引导量 (en_latents_reset2)，默认设为 0.5

        self.image_enhide_latents = None
        self.image_enhide_latents2 = None  # 【新增】独立存放第四步加完水印的潜变量

    def step(self, model_output, timestep, sample, is_forward=True):
        """合并 forward/backward 的推演步骤计算"""
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

        # 主分支预测 (由 lamda_wm 控制)
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]

        if self.image_enhide_latents is not None:
            if self.context_en is not None:
                _, cond_embeddings_en = self.context_en.chunk(2)
            else:
                cond_embeddings_en = cond_embeddings

            # 1. 预测第一分支 (对应 en_latents_reset: 初始隐藏图)
            noise_prediction_wm = self.model.unet(self.image_enhide_latents, t, cond_embeddings_en)["sample"]

            # 基础叠加：无条件 + 主文本项(lamda_wm) + 第一分支隐藏项(lamda_text)
            noise_pred = noise_pred_uncond + \
                         self.lamda_wm * (noise_prediction_text - noise_pred_uncond) + \
                         self.lamda_text * (noise_prediction_wm - noise_pred_uncond)

            # 2. 【新增】预测第二分支 (对应 en_latents_reset2: 第4步带水印隐藏图)
            if self.image_enhide_latents2 is not None:
                noise_prediction_wm2 = self.model.unet(self.image_enhide_latents2, t, cond_embeddings_en)["sample"]
                # 按照用户指定公式规律叠加新分支比重
                noise_pred += self.lamda_wm2 * (noise_prediction_wm2 - noise_pred_uncond)

                # 同步更新第二个 latent 状态
                noise_en_pred2 = noise_pred_uncond + 1.0 * (noise_prediction_wm2 - noise_pred_uncond)
                self.image_enhide_latents2 = self.step(noise_en_pred2, t, self.image_enhide_latents2, is_forward)

            # 同步更新第一个 latent 状态
            noise_en_pred = noise_pred_uncond + 1.0 * (noise_prediction_wm - noise_pred_uncond)
            self.image_enhide_latents = self.step(noise_en_pred, t, self.image_enhide_latents, is_forward)
        else:
            # 回退模式
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
        """对第二提示词 prompt_en 的编码支持"""

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
    def invert(self, arg1, arg2, arg3=None, is_forward=True, mask_latent=None, en_latents_reset=None,
               en_latents_reset2=None):
        """
        参数重载以兼容多种调用方式：
        1. ode.invert(prompt, start_latent, is_forward=True, ...)
        2. ode.invert(prompt_en, prompt, start_latent, is_forward=True, ...)
        """
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

        # 重置分支一
        if en_latents_reset is not None:
            self.image_enhide_latents = en_latents_reset.clone()


        # 重置分支二
        if en_latents_reset2 is not None:
            self.image_enhide_latents2 = en_latents_reset2.clone()


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

    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/11')
    parser.add_argument('--target_bg_path', type=str, default='/opt/data/bing/bldm/utils/experiment/image/17/1_gt.png')

    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--restore_mask_target', type=str, default='white')
    parser.add_argument('--blur_kernel', type=int, default=15)

    parser.add_argument('--private_key', type=str, default='a cat')
    parser.add_argument('--public_key', type=str, default='a tiger')
    parser.add_argument('--bg_private_key', type=str, default='a tiger')
    parser.add_argument('--bg_public_key', type=str, default='on top of a hill overlooking mountains')
    parser.add_argument('--num_steps', type=int, default=50)
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
    # 原始旧逻辑对应：文本引导为主项(=1.0)，其余项为0
    ode.lamda_wm, ode.lamda_text = 1.0, 0.0
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

    # 保存初始潜变量供Step 7使用
    image_hidden_latents_initial = image_hidden_latent.clone()

    # 4. 添加 StegaStamp 水印
    print("4. 正在添加 StegaStamp 水印...")
    watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))
    hide_image_encoded_path = os.path.join(args.save_path, "02_hidden_image_encoded.png")

    # 5. 物理背景替换
    print("5. 正在物理替换背景...")
    image_hide_wm_np, _, _ = load_image(hide_image_encoded_path, resize=True)
    image_target_bg_np, _, _ = load_image(args.target_bg_path, resize=True)

    replace_mask_5 = read_physical_mask(args.mask_path_step5, blur_kernel=args.blur_kernel,
                                        mask_target=args.mask_target)

    bg_replaced_np = np.clip((image_target_bg_np * replace_mask_5 + image_hide_wm_np * (1.0 - replace_mask_5)), 0,
                             255).astype(np.uint8)
    bg_replaced_path = os.path.join(args.save_path, "03_bg_physically_replaced.png")
    cv2.imwrite(bg_replaced_path, cv2.cvtColor(bg_replaced_np, cv2.COLOR_RGB2BGR))

    # ==========================
    # 6. 生成图4：第一阶段全局还原
    # ==========================
    print("6. 全局还原老虎变成山，生成图4...")
    bg_replaced_latent = ode.image2latent(bg_replaced_np)

    # 对应原逻辑的全局还原分配量
    ode.lamda_wm, ode.lamda_text = 1.0, 0.0

    noise_step6 = ode.invert(args.bg_private_key, bg_replaced_latent, is_forward=True)
    revealed_step6_latent = ode.invert(args.bg_public_key, noise_step6, is_forward=False)

    image4_np = ode.latent2image(revealed_step6_latent)
    image4_path = os.path.join(args.save_path, '04_revealed_step6.png')
    cv2.imwrite(image4_path, cv2.cvtColor(image4_np, cv2.COLOR_RGB2BGR))
    print(f"已保存第6步图4至: {image4_path}")

    try:
        acc4 = evaluate_watermark_accuracy(image4_path)
        print(f"--> [检测] 图4的水印提取准确率: {acc4:.4f}")
    except:
        pass

    # ==========================
    # 7. 最终还原：三分支并行加噪清洗
    # ==========================
    print("7. 基于图4执行最终二次掩码还原...")
    image4_latent = ode.image2latent(image4_np)

    # 提取第4步生成的带水印图的 latent
    image_hide_wm_latent = ode.image2latent(image_hide_wm_np)

    # 读取还原 mask6
    restore_mask_latent = read_mask(args.mask_path_step6, invert_mask=(args.restore_mask_target == 'black'),
                                    mask_target=args.mask_target)

    step7_private_key = "a cat"
    step7_public_key = "a tiger"
    bg_public_key = args.bg_public_key

    # 【参数重新映射校准】：
    ode.lamda_wm = 0.1  # 代表 image4_latent 主文本引导比重为 0.5
    ode.lamda_text = 0.7  # 代表 en_latents_reset (初始图项) 比重为 1.0
    ode.lamda_wm2 = 0.1  # 代表 en_latents_reset2 (第4步水印图项) 比重为 0.5

    # 传入对应变量执行前向三分支复合加噪
    noise_step7 = ode.invert(step7_public_key, bg_public_key, image4_latent, is_forward=True,
                             mask_latent=restore_mask_latent,
                             en_latents_reset=image_hidden_latents_initial,
                             en_latents_reset2=image_hide_wm_latent)

    # 最终反向还原（退回单 prompt 引导还原）
    final_reverse_latent = ode.invert(step7_private_key, noise_step7, is_forward=False,
                                      mask_latent=restore_mask_latent)

    final_image = ode.latent2image(final_reverse_latent)
    final_save_path = os.path.join(args.save_path, '05_final_revealed_step7.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"已保存第7步最终还原图至: {final_save_path}")

    # ==========================
    # 最终指标计算
    # ==========================
    psnr_val = calculate_psnr_tensor(image_gt, final_image)
    print(f"流程结束。最终恢复图像 PSNR: {psnr_val:.4f} dB")
    try:
        final_acc = evaluate_watermark_accuracy(final_save_path)
        print(f"最终恢复图的水印提取准确率: {final_acc:.4f}")
    except:
        pass