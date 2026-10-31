import os
import cv2
import torch
import argparse
import numpy as np
import time
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

try:
    from encode_image import StegaStampWatermarker
    from decode_image import evaluate_watermark_accuracy
    from basicsr.metrics import calculate_psnr, calculate_ssim
except ImportError:
    print("警告: 未找到相关模块，请确保文件在当前路径或 PYTHONPATH 中。")

MY_TOKEN = ''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False, steps_offset=1
)

model_path_diffusion = "/data/opt_data_backup/data/stable-diffusion-v1-5"
try:
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_path_diffusion, use_auth_token=MY_TOKEN, scheduler=scheduler,
        torch_dtype=torch.float16
    ).to(device)
    if hasattr(ldm_stable, 'disable_xformers_memory_efficient_attention'):
        ldm_stable.disable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"模型加载警告: {e}。")
    ldm_stable = None

def load_image(image_path, resize=False):
    image = Image.open(image_path).convert("RGB")
    h, w = image.size
    if resize:
        image = image.resize((512, 512))
    else:
        width_padding = -h % 8
        height_padding = -w % 8
        image = ImageOps.expand(image, (0, 0, width_padding, height_padding), fill=None)
    return np.array(image), h, w

def read_physical_mask(mask_paths, dest_size=(512, 512), blur_kernel=15, mask_target='black'):
    paths = mask_paths if isinstance(mask_paths, list) else [mask_paths]
    combined_mask = None
    for path in paths:
        if not os.path.exists(path): continue
        mask = Image.open(path).convert("L").resize(dest_size, Image.BILINEAR)
        mask_np = np.array(mask).astype(np.float32) / 255.0
        combined_mask = mask_np if combined_mask is None else np.maximum(combined_mask, mask_np)
    if combined_mask is None:
        combined_mask = np.ones(dest_size, dtype=np.float32)
    k_size = blur_kernel | 1
    blurred_mask = cv2.GaussianBlur(combined_mask, (k_size, k_size), 0)
    final_mask = blurred_mask if mask_target == 'black' else 1.0 - blurred_mask
    return np.expand_dims(final_mask, axis=-1)

class ConfigurationOptimizer:
    def __init__(self, tau_s=0.5, eps=1e-4, num_steps=50):
        self.tau_s = tau_s
        self.eps = eps
        self.num_steps = num_steps

    def compute_t_s(self, alphas_cumprod):
        snr_t = alphas_cumprod / (1 - alphas_cumprod)
        valid = torch.where(snr_t >= self.tau_s)[0]
        if len(valid) > 0:
            t_s = int(valid[-1])
        else:
            t_s = 0
        t_s = int(t_s * (self.num_steps / 1000))
        t_s = min(max(t_s, 0), 25)
        return t_s

    def compute_gamma(self, delta_ref, s_x):
        rho_ref = torch.sqrt(delta_ref.pow(2).mean(dim=(1, 2), keepdim=True)) / \
                  (torch.sqrt(s_x.pow(2).mean(dim=(1, 2), keepdim=True)) + self.eps)
        gamma = rho_ref.mean().item()
        gamma = max(min(gamma, 2.0), 1.0)
        return gamma

class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        if self.model:
            self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.context_x = None
        self.context_y = None
        self.context_z = None
        self.eps = 1e-4
        self.t_s = 25
        self.gamma = 1.0
        self.beta = 0.9
        self.eta = 0.01

    def step(self, model_output, timestep, sample, is_forward=True):
        step_size = self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        target_t = timestep + step_size if is_forward else timestep - step_size
        target_t = min(max(target_t, -1), 999)
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_target = self.model.scheduler.alphas_cumprod[target_t] if target_t >= 0 else self.model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_target) ** 0.5 * model_output
        return alpha_prod_target ** 0.5 * pred_original_sample + pred_sample_direction

    @torch.no_grad()
    def get_noise_pred(self, latents, t, context, is_forward=False):
        latents = latents.half()
        context = context.half()
        uncond_embeddings, cond_embeddings = context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings).sample
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings).sample
        noise_pred = noise_pred_uncond + (noise_prediction_text - noise_pred_uncond)
        del noise_pred_uncond
        del noise_prediction_text
        return self.step(noise_pred, t, latents, is_forward)

    @torch.no_grad()
    def advanced_acrm_step(self, s_x, s_z, r_x, r_z, t, delta_ref, A_t=None, step_index=None):
        if step_index is None:
            step_index = 0
        s_x = s_x.half()
        s_z = s_z.half()
        r_x = r_x.half()
        r_z = r_z.half()
        delta_ref = delta_ref.half()
        if A_t is not None:
            if A_t.shape[1] == 1:
                A_t = A_t.repeat(1, 4, 1, 1)
            elif A_t.shape[1] == 3:
                A_t = A_t.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
            elif A_t.shape[1] != 4:
                A_t = A_t[:, 0:1, :, :].repeat(1, 4, 1, 1)
            A_t = A_t.half()

        s_x_next = self.get_noise_pred(s_x, t, self.context_x)
        s_z_next = self.get_noise_pred(s_z, t, self.context_z)

        if step_index < self.t_s:
            return s_x_next, s_z_next, r_x, r_z

        s_x_w_next = self.get_noise_pred(torch.clamp(s_x + r_x, -100, 100), t, self.context_x)
        s_z_w_next = self.get_noise_pred(torch.clamp(s_z + r_z, -100, 100), t, self.context_z)

        r_tilde_x = s_x_w_next - s_x_next
        r_tilde_z = s_z_w_next - s_z_next

        rms_r_x = torch.sqrt(r_x.pow(2).mean(dim=(1,2,3), keepdim=True)) + self.eps
        rms_r_tilde_x = torch.sqrt(r_tilde_x.pow(2).mean(dim=(1,2,3), keepdim=True)) + self.eps
        scale_x = rms_r_x / rms_r_tilde_x
        r_tilde_x = r_tilde_x * scale_x

        rms_r_z = torch.sqrt(r_z.pow(2).mean(dim=(1,2,3), keepdim=True)) + self.eps
        rms_r_tilde_z = torch.sqrt(r_tilde_z.pow(2).mean(dim=(1,2,3), keepdim=True)) + self.eps
        scale_z = rms_r_z / rms_r_tilde_z
        r_tilde_z = r_tilde_z * scale_z

        e_ref_x = torch.sqrt(delta_ref.pow(2).mean(dim=(1, 2), keepdim=True)) / (torch.sqrt(s_x.pow(2).mean(dim=(1, 2), keepdim=True)) + self.eps)
        e_ref_z = torch.sqrt(delta_ref.pow(2).mean(dim=(1, 2), keepdim=True)) / (torch.sqrt(s_z.pow(2).mean(dim=(1, 2), keepdim=True)) + self.eps)

        r_x_next = self.beta * r_x + (1 - self.beta) * (self.gamma * e_ref_x * r_tilde_x)
        r_z_next = self.beta * r_z + (1 - self.beta) * (self.gamma * e_ref_z * r_tilde_z)

        r_x_next = torch.nan_to_num(r_x_next, nan=0.0, posinf=0.0, neginf=0.0)
        r_z_next = torch.nan_to_num(r_z_next, nan=0.0, posinf=0.0, neginf=0.0)

        return s_x_next, s_z_next, r_x_next, r_z_next

    @torch.no_grad()
    def latent2image(self, latents):
        latents = latents.to(torch.float16)
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = (image / 2 + 0.5).clamp(0, 1)
        return (image.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    @torch.no_grad()
    def image2latent(self, image):
        if isinstance(image, Image.Image): image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4: return image
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        image = image.to(torch.float16)
        return (self.model.vae.encode(image)['latent_dist'].mean * 0.18215).to(torch.float16)

    @torch.no_grad()
    def init_prompt(self, p_x: str, p_y: str, p_z: str = None):
        def encode_text(p):
            uncond_input = self.model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            text_input = self.model.tokenizer([p], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            return torch.cat([
                self.model.text_encoder(uncond_input.input_ids.to(device))[0],
                self.model.text_encoder(text_input.input_ids.to(device))[0]
            ]).to(torch.float16)
        self.context_x = encode_text(p_x)
        self.context_y = encode_text(p_y)
        self.context_z = encode_text(p_z) if p_z else self.context_y

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gt.png')
    parser.add_argument('--stegastamp_model', type=str, default='/data/opt_data_backup/data/140000.pth')
    parser.add_argument('--mask_path_step5', nargs='+', default=["/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/seg_output/1/mask_002.png"])
    parser.add_argument('--mask_path_step6', nargs='+', default=["/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/seg_output/3/mask_003.png", "/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/seg_output/3/mask_024.png"])
    parser.add_argument('--save_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gauss/no_rnp')
    parser.add_argument('--target_bg_path', type=str, default='/data/opt_data_backup/data/target_bg.png')
    parser.add_argument('--mask_target', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--blur_kernel', type=int, default=15)
    parser.add_argument('--private_key', type=str, default='a cat')
    parser.add_argument('--public_key', type=str, default='a tiger')
    parser.add_argument('--bg_private_key', type=str, default='a tiger')
    parser.add_argument('--bg_public_key', type=str, default='on top of a hill overlooking mountains')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.75, help='[保留参数但不使用]')
    parser.add_argument('--ts', type=int, default=None, help='手动指定激活时间步')
    parser.add_argument('--gamma', type=float, default=None, help='手动指定缩放因子')
    parser.add_argument('--beta', type=float, default=0.9, help='残差连接系数')
    parser.add_argument('--disable_jrt', action='store_true', help='关闭潜空间JRT')
    # ===== 新增：跳过水印，用于 PSNR 消融 =====
    parser.add_argument('--no_watermark', action='store_true',
                        help='消融：第4步跳过水印，得到无水印版结果')
    # ===== 新增结束 =====
    parser.add_argument('--ref_pinjie_path', type=str,
                        default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gauss/14/nojrt_pinjie.png',
                        help='参考 08_pinjie，用于 PSNR 对比')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    ode = ODESolve(ldm_stable, args.num_steps)
    ode.beta = args.beta
    watermarker = StegaStampWatermarker(args.stegastamp_model, device=device)

    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "02_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    image_gt_latent = ode.image2latent(image_gt)

    print("3. 正在生成 Hidden Image...")
    ode.init_prompt(args.private_key, args.public_key)
    latent_noise = image_gt_latent.clone()
    for i in tqdm(range(ode.num_ddim_steps), leave=False):
        t = ldm_stable.scheduler.timesteps[len(ldm_stable.scheduler.timesteps) - i - 1]
        latent_noise = ode.get_noise_pred(latent_noise, t, ode.context_x, is_forward=True)

    image_hidden_latent = latent_noise.clone()
    for i in tqdm(range(ode.num_ddim_steps), leave=False):
        t = ldm_stable.scheduler.timesteps[i]
        image_hidden_latent = ode.get_noise_pred(image_hidden_latent, t, ode.context_y, is_forward=False)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "03_hidden_image.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

    # ===== 改动：第4步根据 --no_watermark 分支 =====
    if args.no_watermark:
        print("4. [消融] 跳过水印，直接用 hidden image")
        image_hide_wm_np = image_hidden_np.copy()
        image_hide_wm_latent = image_hidden_latent.clone()
    else:
        print("4. 正在添加水印...")
        watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))
        hide_image_encoded_path = os.path.join(args.save_path, "03_hidden_image_encoded.png")
        image_hide_wm_np, _, _ = load_image(hide_image_encoded_path, resize=True)
        image_hide_wm_latent = ode.image2latent(image_hide_wm_np)
    # ===== 改动结束 =====

    print("5. 正在构造区域替换后的图像...")
    image_target_bg_np, _, _ = load_image(args.target_bg_path, resize=True)
    mask6_np = read_physical_mask(args.mask_path_step6, blur_kernel=args.blur_kernel, mask_target=args.mask_target)
    Zw_prime_np = np.clip((image_hide_wm_np * mask6_np + image_target_bg_np * (1.0 - mask6_np)), 0, 255).astype(np.uint8)
    Zw_prime_latent = ode.image2latent(Zw_prime_np)

    print("6. 正在反演纯背景轨迹 Z...")
    image_z_latent = ode.image2latent(image_target_bg_np)
    z_z_current = image_z_latent.clone()
    ode.init_prompt(p_x=args.private_key, p_y=args.public_key, p_z=args.bg_private_key)
    for i in tqdm(range(ode.num_ddim_steps), desc="Caching Pure BG Inversion"):
        t = ldm_stable.scheduler.timesteps[len(ldm_stable.scheduler.timesteps) - i - 1]
        z_z_current = ode.get_noise_pred(z_z_current, t, ode.context_z, is_forward=True)
    z_z_T = z_z_current.clone()

    print("7. 正在反演原始主体轨迹 X...")
    x_current = image_gt_latent.clone()
    for i in tqdm(range(ode.num_ddim_steps), desc="Caching X Inversion"):
        t = ldm_stable.scheduler.timesteps[len(ldm_stable.scheduler.timesteps) - i - 1]
        x_current = ode.get_noise_pred(x_current, t, ode.context_x, is_forward=True)
    x_T = x_current.clone()

    print("8. 预先计算并缓存前向 Inversion 轨迹...")
    z_y_current = image_hidden_latent.clone()
    z_yw_current = image_hide_wm_latent.clone()
    z_zw_current = Zw_prime_latent.clone()

    cached_timesteps = ldm_stable.scheduler.timesteps
    for i in tqdm(range(ode.num_ddim_steps), desc="Caching Forward Inversion"):
        t = cached_timesteps[len(cached_timesteps) - i - 1]
        z_y_current = ode.get_noise_pred(z_y_current, t, ode.context_x, is_forward=True)
        z_yw_current = ode.get_noise_pred(z_yw_current, t, ode.context_x, is_forward=True)
        z_zw_current = ode.get_noise_pred(z_zw_current, t, ode.context_z, is_forward=True)

    z_y_T = z_y_current.clone()
    z_yw_T = z_yw_current.clone()
    z_zw_T = z_zw_current.clone()

    delta_ref = z_yw_T - z_y_T
    r_x = delta_ref * 1.0
    r_z = delta_ref * 1.0

    # ===== 改动：R_np 和 A_t 在无水下处理除零 =====
    R_np = image_hide_wm_np.astype(np.float32) - image_hidden_np.astype(np.float32)
    R_np_u8 = np.clip(R_np, 0, 255).astype(np.uint8)
    A_np_gray = cv2.cvtColor(R_np_u8, cv2.COLOR_RGB2GRAY) if R_np_u8.ndim == 3 else R_np_u8
    A_np_gray = cv2.GaussianBlur(A_np_gray, (15, 15), 0)
    a_max = A_np_gray.max()
    if a_max < 1e-6:
        A_np_gray = np.ones_like(A_np_gray) * 0.5
    else:
        A_np_gray = A_np_gray / a_max
    A_t_resized = cv2.resize(A_np_gray, (64, 64))
    A_t = torch.from_numpy(A_t_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    A_t = A_t.repeat(1, 4, 1, 1)
    # ===== 改动结束 =====

    config_opt = ConfigurationOptimizer(tau_s=0.5, num_steps=args.num_steps)
    alphas_cumprod = ldm_stable.scheduler.alphas_cumprod

    if args.ts is not None:
        ode.t_s = args.ts
        print(f"--> 手动指定 t_s = {ode.t_s}")
    else:
        ode.t_s = config_opt.compute_t_s(alphas_cumprod)
        print(f"--> 使用解析估计器 t_s = {ode.t_s}")

    if args.gamma is not None:
        ode.gamma = args.gamma
        print(f"--> 手动指定 gamma = {ode.gamma:.4f}")
    else:
        ode.gamma = config_opt.compute_gamma(delta_ref, z_y_T)
        print(f"--> 使用解析估计器 gamma = {ode.gamma:.4f}")

    print(f"--> [去掉 R_np 加法] 最终输出 = 08_pinjie，不加像素残差")

    print("10. 正在进行雅可比引导水印残差运输...")
    s_x = x_T.clone()
    s_z = z_z_T.clone()

    if args.disable_jrt:
        r_x = torch.zeros_like(r_x)
        r_z = torch.zeros_like(r_z)
        print("--> [消融] 已关闭潜空间 JRT（去噪仍正常执行）")

    for t_idx in range(ode.num_ddim_steps):
        t = cached_timesteps[t_idx]
        if args.disable_jrt:
            s_x = ode.get_noise_pred(s_x, t, ode.context_x)
            s_z = ode.get_noise_pred(s_z, t, ode.context_z)
        else:
            s_x, s_z, r_x, r_z = ode.advanced_acrm_step(
                s_x, s_z, r_x, r_z, t, delta_ref, A_t=A_t, step_index=t_idx
            )
            if t_idx % 5 == 0:
                norm_x = torch.sqrt(r_x.pow(2).mean()).item()
                norm_z = torch.sqrt(r_z.pow(2).mean()).item()
                print(f"step {t_idx}, t={t}, ||r_x|| = {norm_x:.4f}, ||r_z|| = {norm_z:.4f}")

    final_latent_x = s_x + r_x
    final_latent_z = s_z + r_z

    image4_np_x = ode.latent2image(final_latent_x)
    image4_np_z = ode.latent2image(final_latent_z)

    image4_path = os.path.join(args.save_path, '06_revealed_step6.png')
    cv2.imwrite(image4_path, cv2.cvtColor(image4_np_x, cv2.COLOR_RGB2BGR))

    image4_x_path = os.path.join(args.save_path, '06a_subject.png')
    cv2.imwrite(image4_x_path, cv2.cvtColor(image4_np_x, cv2.COLOR_RGB2BGR))
    image4_z_path = os.path.join(args.save_path, '06b_background.png')
    cv2.imwrite(image4_z_path, cv2.cvtColor(image4_np_z, cv2.COLOR_RGB2BGR))

    try:
        acc_subject = evaluate_watermark_accuracy(image4_x_path)
        acc_background = evaluate_watermark_accuracy(image4_z_path)
    except Exception as e:
        print(f"分支解码失败: {e}")
        acc_subject = acc_background = 0.0
    print(f"[06a Subject Bit Accuracy] : {acc_subject:.4f}")
    print(f"[06b Background Bit Accuracy]: {acc_background:.4f}")

    print("11. 提取指定形状并进行区域拼接...")
    final_image = np.clip((image4_np_x * mask6_np + image4_np_z * (1.0 - mask6_np)), 0, 255).astype(np.uint8)
    pinjie_save_path = os.path.join(args.save_path, '08_pinjie.png')
    cv2.imwrite(pinjie_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    print("12. [去掉 R_np 加法] 最终输出 = 08_pinjie（纯 JRT 运输结果）")
    subtracted_clipped_np = final_image.copy()
    subtracted_save_path = os.path.join(args.save_path, "09_subtracted.png")
    cv2.imwrite(subtracted_save_path, cv2.cvtColor(subtracted_clipped_np, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 60)
    try:
        ref_img = cv2.imread(args.ref_pinjie_path, cv2.IMREAD_COLOR)
        if ref_img is None:
            raise FileNotFoundError(f"参考图不存在: {args.ref_pinjie_path}")
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.resize(ref_img, (512, 512))
        final_psnr = calculate_psnr(ref_img, subtracted_clipped_np,
                                     crop_border=0, input_order='HWC', test_y_channel=False)
        print(f"[09 PSNR vs ref_pinjie] : {final_psnr:.4f} dB")
    except Exception as e:
        final_psnr = 0.0
        print(f"PSNR 计算出错: {e}")

    try:
        acc_step6 = evaluate_watermark_accuracy(image4_path)
        acc_pinjie = evaluate_watermark_accuracy(pinjie_save_path)
        acc_subtracted = evaluate_watermark_accuracy(subtracted_save_path)
    except Exception as e:
        print(f"水印检测失败: {e}")
        acc_step6 = acc_pinjie = acc_subtracted = 0.0

    end_time = time.time()
    total_time = end_time - start_time

    print(f"[Total Attack Time]     : {total_time:.2f} s")
    print(f"[08 Bit Accuracy]       : {acc_pinjie:.4f}")
    print(f"[09 Bit Accuracy]       : {acc_subtracted:.4f}")
    print(f"[PSNR vs ref]           : {final_psnr:.4f} dB")
    print(f"[06 Bit Accuracy]       : {acc_step6:.4f}")
    print("=" * 60 + "\n")

    result_file = os.path.join(args.save_path, "results.txt")
    with open(result_file, "w") as f:
        f.write(f"Total Attack Time: {total_time:.2f} s\n")
        f.write(f"06 (JRT only): {acc_step6:.4f}\n")
        f.write(f"08 (Composition): {acc_pinjie:.4f}\n")
        f.write(f"09 (No R_np): {acc_subtracted:.4f}\n")
        f.write(f"06a Subject: {acc_subject:.4f}\n")
        f.write(f"06b Background: {acc_background:.4f}\n")
        f.write(f"09 PSNR vs ref: {final_psnr:.4f} dB\n")
        f.write(f"Parameters: no_watermark={args.no_watermark}, disable_jrt={args.disable_jrt}, beta={args.beta}, ts={ode.t_s}, gamma={ode.gamma:.4f}\n")
    print(f"结果已保存至: {result_file}")