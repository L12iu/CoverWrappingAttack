from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
import lpips

# 使用 bayes_opt 确保和 advance.py 统一
from bayes_opt import BayesianOptimization

# =========================================================================
# 1. 内置 PSNR / SSIM 计算
# =========================================================================
try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if ssim_fn is None:
        return 0.0
    return float(ssim_fn(img1, img2, channel_axis=2, data_range=255))


# =========================================================================
# 2. TrustMark 核心模型定义
# =========================================================================
class TrustMarkEncoder(nn.Module):
    def __init__(self, message_length=64, channels=3):
        super(TrustMarkEncoder, self).__init__()
        self.message_length = message_length

        self.msg_fc = nn.Sequential(
            nn.Linear(message_length, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, image, message):
        B, C, H, W = image.shape
        x_feat = self.conv1(image)
        x_feat = self.conv2(x_feat)

        m_feat = self.msg_fc(message)
        m_feat = m_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

        combined = torch.cat([x_feat, m_feat], dim=1)
        residual = self.fuse_conv(combined) * 0.1
        watermarked_image = torch.clamp(image + residual, 0.0, 1.0)
        return watermarked_image


class TrustMarkDecoder(nn.Module):
    def __init__(self, message_length=64, channels=3):
        super(TrustMarkDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, message_length)
        )

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)

        pooled = self.global_pool(feat).squeeze(-1).squeeze(-1)
        logits = self.fc(pooled)
        return logits


class TrustMarkWatermarker:
    def __init__(self, encoder_ckpt: str = None, decoder_ckpt: str = None, msg_length=64, device='cuda'):
        self.device = device
        self.msg_length = msg_length
        self.encoder = TrustMarkEncoder(message_length=msg_length).to(device)
        self.decoder = TrustMarkDecoder(message_length=msg_length).to(device)

        if encoder_ckpt and os.path.exists(encoder_ckpt):
            self.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
            print(f"[TrustMark] 成功加载 Encoder 权重: {encoder_ckpt}")
        else:
            print("[TrustMark] 警告: 未找到 Encoder 权重文件，将使用随机初始化网络测试。")

        if decoder_ckpt and os.path.exists(decoder_ckpt):
            self.decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
            print(f"[TrustMark] 成功加载 Decoder 权重: {decoder_ckpt}")
        else:
            print("[TrustMark] 警告: 未找到 Decoder 权重文件，将使用随机初始化网络测试。")

        self.encoder.eval()
        self.decoder.eval()

    def add_watermark(self, image_np: np.ndarray, secret_msg: torch.Tensor) -> np.ndarray:
        img_t = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)
        secret_msg = secret_msg.to(self.device)

        with torch.no_grad():
            wm_img_t = self.encoder(img_t, secret_msg)
            wm_img_np = (wm_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        return wm_img_np

    def evaluate_watermark_accuracy(self, image_np: np.ndarray, secret_msg: torch.Tensor) -> float:
        img_t = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)
        secret_msg = secret_msg.to(self.device)

        with torch.no_grad():
            logits = self.decoder(img_t)
            pred_bits = (torch.sigmoid(logits) > 0.5).float()
            correct = (pred_bits == secret_msg).float().sum(dim=1)
            bit_acc = (correct / secret_msg.size(1)).mean().item()

        return bit_acc


# =========================================================================
# 3. Diffusion 模型与环境配置
# =========================================================================
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False
)
MY_TOKEN = os.getenv('HF_TOKEN', None)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model_path_diffusion = os.getenv('CROSS_SD_MODEL', '/data/opt_data_backup/data/stable-diffusion-v1-5')
ldm_stable = StableDiffusionPipeline.from_pretrained(
    model_path_diffusion,
    token=MY_TOKEN,
    scheduler=scheduler
).to(device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    pass


# =========================================================================
# 4. 辅助函数与严格控制物理意义的 ODE Solver
# =========================================================================
def load_image(image_path, resize=True):
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


def calculate_lpips_metric(img1_np, img2_np, loss_fn, device_obj):
    t1 = torch.from_numpy(img1_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    t2 = torch.from_numpy(img2_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        lpips_val = loss_fn(t1.to(device_obj), t2.to(device_obj)).item()
    return lpips_val


class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)  # 严格遵守外部步数控制
        self.prompt = None
        self.context = None
        self.lamda_text = 1.0
        self.lamda_wm = 0.5
        self.image_enhide_latents = None

    def prev_step(self, model_output, timestep, sample):
        """逆向去噪步骤 (t -> t-1)"""
        t_val = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        prev_timestep = t_val - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[t_val]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        """正向加噪反演步骤 (t -> t+1)"""
        t_val = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        timestep_clamped = min(t_val - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps,
                               999)
        next_timestep = t_val

        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep_clamped] if timestep_clamped >= 0 else self.scheduler.final_alpha_cumprod
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

            # 在噪声预测层面进行细致的融合
            noise_pred = noise_pred_uncond + \
                         w_text * (noise_prediction_text - noise_pred_uncond) + \
                         w_wm * (noise_prediction_wm - noise_pred_uncond)

            noise_en_pred = noise_pred_uncond + 1 * (noise_prediction_wm - noise_pred_uncond)

            if is_forward:
                self.image_enhide_latents = self.next_step(noise_en_pred, t, self.image_enhide_latents)
            else:
                self.image_enhide_latents = self.prev_step(noise_en_pred, t, self.image_enhide_latents)
        else:
            w_text = self.lamda_text
            noise_pred = noise_pred_uncond + w_text * (noise_prediction_text - noise_pred_uncond)

        # 严格遵守正反向的步长求解推导
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
    def ddim_loop(self, latent, is_forward=True):
        all_latent = [latent]
        latent = latent.clone().detach()

        for i in tqdm(range(self.num_ddim_steps), leave=False):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            all_latent.append(latent)

        return all_latent

    def invert(self, prompt, start_latent, is_forward, en_latents_reset=None):
        self.init_prompt(prompt)

        # ⚠️ 【极其致命的缺陷已修复】：确保纯净无水印生成轨迹不会被残留状态污染
        if en_latents_reset is not None:
            self.image_enhide_latents = en_latents_reset.clone()
        else:
            self.image_enhide_latents = None  # 阻断上一轮的水印潜变量残留！

        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]


# =========================================================================
# 5. 改进后的贝叶斯优化函数 (Decoder-Free Objective)
# =========================================================================
def optimize_lamda_trustmark_2d(ode: ODESolve,
                                latent_input,
                                enhide_latents_initial,
                                image_gt,
                                prompt_private: str,
                                prompt_public: str,
                                trustmark_watermarker: TrustMarkWatermarker,
                                secret_msg: torch.Tensor,
                                gamma: float = 1.0,
                                bo_init_points: int = 3,
                                bo_n_iter: int = 7):
    """
    Decoder-free Bayesian Optimization
    使用 L_sem (Eq. 11) 和 L_pres (Eq. 14) 作为代理损失，避免调用解码器。
    """
    print("启动 Configuration Optimizer (Bayesian Optimization)...")

    delta_z_ref = latent_input - enhide_latents_initial
    noise_input = ode.invert(prompt_public, latent_input, is_forward=True, en_latents_reset=enhide_latents_initial)
    X_np_norm = image_gt.astype(np.float32) / 255.0

    def objective(lam_text, lam_wm):
        # A. 带水印轨迹
        ode.lamda_text = float(lam_text)
        ode.lamda_wm = float(lam_wm)
        test_revealed_latent = ode.invert(prompt_private, noise_input, is_forward=False, en_latents_reset=noise_input)
        test_image_np = ode.latent2image(test_revealed_latent, return_type='np').astype(np.float32)

        # B. 纯净参考轨迹 (此处如果缺失 else: None 的修复，会导致纯净轨迹受损)
        ode.lamda_text = float(lam_text)
        clean_revealed_latent = ode.invert(prompt_private, noise_input, is_forward=False, en_latents_reset=None)

        # C. 损失计算
        test_img_norm = test_image_np / 255.0
        loss_sem = np.linalg.norm(X_np_norm - test_img_norm) ** 2

        delta_z_rec = test_revealed_latent - clean_revealed_latent
        loss_pres = torch.norm(delta_z_rec - delta_z_ref).item() ** 2

        objective_cost = loss_sem + gamma * loss_pres
        return -objective_cost

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={'lam_text': (0.0, 1.0), 'lam_wm': (0.0, 1.0)},
        random_state=42,
        allow_duplicate_points=True
    )

    optimizer.maximize(init_points=bo_init_points, n_iter=bo_n_iter)
    best_config = optimizer.max['params']

    print(f"--> Bayesian Optimization 最佳配置: λ_text={best_config['lam_text']:.4f}, λ_wm={best_config['lam_wm']:.4f}")
    return best_config['lam_text'], best_config['lam_wm']


# =========================================================================
# 6. 主程序
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustMark Cover Warping Attack Pipeline")
    parser.add_argument('--gt_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/asserts/1.png')
    parser.add_argument('--trustmark_encoder_ckpt', type=str, default='')
    parser.add_argument('--trustmark_decoder_ckpt', type=str, default='')
    parser.add_argument('--private_key', type=str, default='Eiffel tower')
    parser.add_argument('--public_key', type=str, default='a bottle')
    parser.add_argument('--save_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/trustmark_test')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--msg_length', type=int, default=64)
    parser.add_argument('--tau_w', type=float, default=0.85)
    parser.add_argument('--tau_psnr', type=float, default=20.0)
    parser.add_argument('--tau_lpips', type=float, default=0.20)

    # 贝叶斯优化对应参数
    parser.add_argument('--gamma', type=float, default=1.0, help="控制语义重构与水印保存的折中系数")
    parser.add_argument('--bo_init_points', type=int, default=3, help="贝叶斯优化初始探索点数")
    parser.add_argument('--bo_n_iter', type=int, default=7, help="贝叶斯优化迭代次数")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("正在初始化 Diffusion 模型与 TrustMark 水印系统...")
    ode = ODESolve(ldm_stable, args.num_steps)
    watermarker = TrustMarkWatermarker(
        encoder_ckpt=args.trustmark_encoder_ckpt,
        decoder_ckpt=args.trustmark_decoder_ckpt,
        msg_length=args.msg_length,
        device=device
    )
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)

    torch.manual_seed(1234)
    secret_msg = torch.randint(0, 2, (1, args.msg_length)).float().to(device)

    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "01_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    image_gt_latent = ode.image2latent(image_gt)

    print("正在生成 Hidden Image (Diffusion 过程)...")
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)
    ode.lamda_text = 1.0
    ode.lamda_wm = 0.0
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False)
    image_hidden_np = ode.latent2image(image_hidden_latent)

    hidden_path = os.path.join(args.save_path, "02_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))
    print(f"Hidden Image 已生成并保存至: {hidden_path}")

    print("正在为 Hidden Image 嵌入 TrustMark 水印...")
    image_hide_np = watermarker.add_watermark(image_hidden_np, secret_msg)
    hide_image_path = os.path.join(args.save_path, "02_hidden_image_encoded.png")
    cv2.imwrite(hide_image_path, cv2.cvtColor(image_hide_np, cv2.COLOR_RGB2BGR))

    image_hide_latent = ode.image2latent(image_hide_np)
    image_hidden_latents_initial = image_hidden_latent.clone()

    best_text_w, best_wm_w = optimize_lamda_trustmark_2d(
        ode=ode,
        latent_input=image_hide_latent,
        enhide_latents_initial=image_hidden_latents_initial,
        image_gt=image_gt,
        prompt_private=args.private_key,
        prompt_public=args.public_key,
        trustmark_watermarker=watermarker,
        secret_msg=secret_msg,
        gamma=args.gamma,
        bo_init_points=args.bo_init_points,
        bo_n_iter=args.bo_n_iter
    )

    ode.lamda_text = float(best_text_w)
    ode.lamda_wm = float(best_wm_w)

    final_latent_noise = ode.invert(args.public_key, image_hide_latent, is_forward=True,
                                    en_latents_reset=image_hidden_latents_initial)
    final_reverse_latent = ode.invert(args.private_key, final_latent_noise, is_forward=False)
    final_image = ode.latent2image(final_reverse_latent)

    final_save_path = os.path.join(args.save_path, '03_best_revealed.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    final_acc = watermarker.evaluate_watermark_accuracy(final_image, secret_msg)
    final_psnr = calculate_psnr(image_gt, final_image)
    final_ssim = calculate_ssim(image_gt, final_image)
    final_lpips = calculate_lpips_metric(image_gt, final_image, loss_fn_vgg, device)
    您指出的这三个隐患非常精准且致命！这完全切中了扩散模型（Diffusion
    Models）在做
    DDIM
    反演和特征引导时最容易踩坑的核心原理：

    1. ** 步数错乱 ** 会破坏贝叶斯优化和加噪 / 去噪的时间调度（Schedule）。
    2. ** 正反向逻辑混淆 ** 会直接导致潜变量（Latent）不仅没有回到正确的噪声分布，反而变成无意义的乱码。
    3. ** 潜空间硬相加 ** 会破坏图像的流形结构（Manifold），而通过
    `noise_pred`
    进行引导（Guidance）才是符合分类器无引导（Classifier - Free
    Guidance, CFG）和特征注入的正确做法。

    由于我当前无法直接读取我们上一轮对话中那份完整的
    `TrustMark`（以
    `
    from typing import Optional

    ...
    ` 开头）的代码文件，为了最快、最安全地帮您修复，我将这三个关键逻辑的 ** 正确代码片段 ** 梳理如下。您可以直接将它们替换回您手头那份正确的代码中，或者 ** 如果您方便的话，可以将那份原始代码粘贴给我，我来为您完成最终的无缝合并。 **

    ### 核心修复代码片段参考

    #### 1. 修复 `ddim_loop` (解决隐患1 和 隐患2)
    在循环中接收
    `num_inference_steps`，并严格区分
    `is_forward`
    的
    `next_step`（加噪）和
    `prev_step`（去噪）。

    ```python


    def ddim_loop(self, latent: torch.Tensor, is_forward: bool, num_inference_steps: int = 50) -> List[torch.Tensor]:
        # ✅ 修复1：使用传入的推理步数，而不是 1000 步
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 如果是正向反演 (加噪)，需要将时间步反转 (从 t=0 向 t=T)
        if is_forward:
            timesteps = reversed(timesteps)

        latent_list = [latent]

        for i, t in enumerate(timesteps):
            # ✅ 修复3：通过自定义的 get_noise_pred 在噪声预测层面进行融合
            noise_pred = self.get_noise_pred(latent, t, is_forward=is_forward)

            # ✅ 修复2：严格区分正向加噪和逆向去噪的解码方法
            if is_forward:
                latent = self.next_step(noise_pred, t, latent)
            else:
                latent = self.prev_step(noise_pred, t, latent)

            latent_list.append(latent)

        return latent_list