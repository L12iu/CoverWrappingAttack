import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np

from typing import Union, List

from diffusers import StableDiffusionPipeline, DDIMScheduler

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
    parser.add_argument('--test_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/14/08_pinjie.png')
    parser.add_argument('--stegastamp_model', type=str,
                        default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth')
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/13')
    parser.add_argument('--private_key', type=str, default='a cat')
    parser.add_argument('--public_key', type=str, default='a tiger')
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

    # 3. 生成 Hidden Image (图片 A)
    print("3. 正在生成 Hidden Image (图片A)...")
    ode.lamda_text, ode.lamda_wm = 1.0, 0.0
    latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)
    image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False)

    image_hidden_np = ode.latent2image(image_hidden_latent)
    hidden_path = os.path.join(args.save_path, "02_hidden_image_A.png")
    cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

    # 4. 生成编码图 (图片 D2)
    print("4. 正在添加 StegaStamp 水印-02_hidden_image_A_encoded...")
    watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))

    # 假设 D2 图片的固定读取路径
    D2_path = '/opt/data/jiang/cross/CRoSS/image_jiang/gauss/9/02_hidden_image_residual.png'

    # =========================================================================
    # 5. 【核心步骤】生成B并计算残差 D1 和 D1-D2
    # =========================================================================
    print("5. 正在生成图片B，计算残差 D1 以及 D1-D2...")

    test_prompt = "tiger"
    ode.lamda_text, ode.lamda_wm = 1.0, 0.0

    noise_A = ode.invert(test_prompt, image_hidden_latent, is_forward=True)
    latent_B = ode.invert(test_prompt, noise_A, is_forward=False)
    image_B_np = ode.latent2image(latent_B)

    # 保存图片 B
    image_b_path = os.path.join(args.save_path, "02_extra_image_B.png")
    cv2.imwrite(image_b_path, cv2.cvtColor(image_B_np, cv2.COLOR_RGB2BGR))

    # 计算残差 D1 = B - A (使用 float32 防止溢出)
    D1 = image_B_np.astype(np.float32) - image_hidden_np.astype(np.float32)

    # 读取 D2 图片
    if not os.path.exists(D2_path):
        print(f"【错误】未找到 D2 图片，请检查路径: {D2_path}")
    else:
        image_D2_np, _, _ = load_image(D2_path, resize=True)

        # 计算 D1 - D2
        D1_minus_D2 = D1 - image_D2_np.astype(np.float32)

        # 归一化/截断到 0-255 保存可视化结果
        D1_minus_D2_visual = np.clip(np.abs(D1_minus_D2), 0, 255).astype(np.uint8)
        d1_minus_d2_save_path = os.path.join(args.save_path, "02_extra_D1_minus_D2.png")
        cv2.imwrite(d1_minus_d2_save_path, cv2.cvtColor(D1_minus_D2_visual, cv2.COLOR_RGB2BGR))

        print(f"--> (D1-D2) 平均绝对误差: {np.mean(np.abs(D1_minus_D2)):.4f}")
        print(f"--> 已将 (D1-D2) 保存至: {d1_minus_d2_save_path}")

        # =========================================================================
        # 6. 用 GT 减去 (D1-D2) 残差，生成测试图片
        # =========================================================================
        print("\n6. 正在用 GT 减去 (D1-D2) 残差，生成反向扰动图片...")
        gt_float = image_gt.astype(np.float32)
        residual_float = D1_minus_D2.astype(np.float32)

        subtracted_np = gt_float - residual_float
        subtracted_clipped_np = np.clip(subtracted_np, 0, 255).astype(np.uint8)

        subtracted_save_path = os.path.join(args.save_path, "03_gt_minus_residual.png")
        cv2.imwrite(subtracted_save_path, cv2.cvtColor(subtracted_clipped_np, cv2.COLOR_RGB2BGR))
        print(f"--> 已将相减后的图片保存至: {subtracted_save_path}")

        # =========================================================================
        # 7. 测试 GT 相减后图片的水印准确率
        # =========================================================================
        print("\n7. 正在对 GT 相减后的图片进行水印准确率检测...")
        try:
            subtracted_acc = evaluate_watermark_accuracy(subtracted_save_path)
            print("=" * 70)
            print(f"   [GT - 残差] 图片的水印提取准确率: {subtracted_acc:.4f}")
            print("=" * 70)
        except Exception as e:
            print(f"   [检测失败] 无法在 GT 相减图片上提取水印，报错信息: {e}")
        import os
        import cv2
        import torch
        import argparse
        import numpy as np
        from PIL import Image, ImageOps
        from tqdm import tqdm
        import numpy as np

        from typing import Union, List

        from diffusers import StableDiffusionPipeline, DDIMScheduler

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
            parser.add_argument('--test_path', type=str,
                                default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/14/08_pinjie.png')
            parser.add_argument('--stegastamp_model', type=str,
                                default='/opt/data/helin/Code/StegaStamp_pytorch/checkpoints/Test01/140000.pth')
            parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/gauss/13')
            parser.add_argument('--private_key', type=str, default='a cat')
            parser.add_argument('--public_key', type=str, default='a tiger')
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

            # 3. 生成 Hidden Image (图片 A)
            print("3. 正在生成 Hidden Image (图片A)...")
            ode.lamda_text, ode.lamda_wm = 1.0, 0.0
            latent_noise = ode.invert(args.private_key, image_gt_latent, is_forward=True)
            image_hidden_latent = ode.invert(args.public_key, latent_noise, is_forward=False)

            image_hidden_np = ode.latent2image(image_hidden_latent)
            hidden_path = os.path.join(args.save_path, "02_hidden_image_A.png")
            cv2.imwrite(hidden_path, cv2.cvtColor(image_hidden_np, cv2.COLOR_RGB2BGR))

            # 4. 生成编码图 (图片 D2)
            print("4. 正在添加 StegaStamp 水印-02_hidden_image_A_encoded...")
            watermarker.add_watermark(hidden_path, save_dir=args.save_path, output_size=(512, 512))

            # 假设 D2 图片的固定读取路径
            D2_path = '/opt/data/jiang/cross/CRoSS/image_jiang/gauss/9/02_hidden_image_residual.png'

            # =========================================================================
            # 5. 【核心步骤】生成B并计算残差 D1 和 D1-D2
            # =========================================================================
            print("5. 正在生成图片B，计算残差 D1 以及 D1-D2...")

            test_prompt = "tiger"
            ode.lamda_text, ode.lamda_wm = 1.0, 0.0

            noise_A = ode.invert(test_prompt, image_hidden_latent, is_forward=True)
            latent_B = ode.invert(test_prompt, noise_A, is_forward=False)
            image_B_np = ode.latent2image(latent_B)

            # 保存图片 B
            image_b_path = os.path.join(args.save_path, "02_extra_image_B.png")
            cv2.imwrite(image_b_path, cv2.cvtColor(image_B_np, cv2.COLOR_RGB2BGR))

            # 计算残差 D1 = B - A (使用 float32 防止溢出)
            D1 = image_B_np.astype(np.float32) - image_hidden_np.astype(np.float32)

            # 读取 D2 图片
            if not os.path.exists(D2_path):
                print(f"【错误】未找到 D2 图片，请检查路径: {D2_path}")
            else:
                image_D2_np, _, _ = load_image(D2_path, resize=True)

                # 计算 D1 - D2
                D1_minus_D2 = D1 - image_D2_np.astype(np.float32)

                # 归一化/截断到 0-255 保存可视化结果
                D1_minus_D2_visual = np.clip(np.abs(D1_minus_D2), 0, 255).astype(np.uint8)
                d1_minus_d2_save_path = os.path.join(args.save_path, "02_extra_D1_minus_D2.png")
                cv2.imwrite(d1_minus_d2_save_path, cv2.cvtColor(D1_minus_D2_visual, cv2.COLOR_RGB2BGR))

                print(f"--> (D1-D2) 平均绝对误差: {np.mean(np.abs(D1_minus_D2)):.4f}")
                print(f"--> 已将 (D1-D2) 保存至: {d1_minus_d2_save_path}")

                # =========================================================================
                # 6. 用 GT 减去 (D1-D2) 残差，生成测试图片
                # =========================================================================
                print("\n6. 正在用 GT 减去 (D1-D2) 残差，生成反向扰动图片...")
                gt_float = image_gt.astype(np.float32)
                residual_float = D1_minus_D2.astype(np.float32)

                subtracted_np = gt_float - residual_float
                subtracted_clipped_np = np.clip(subtracted_np, 0, 255).astype(np.uint8)

                subtracted_save_path = os.path.join(args.save_path, "03_gt_minus_residual.png")
                cv2.imwrite(subtracted_save_path, cv2.cvtColor(subtracted_clipped_np, cv2.COLOR_RGB2BGR))
                print(f"--> 已将相减后的图片保存至: {subtracted_save_path}")

                # =========================================================================
                # 7. 测试 GT 相减后图片的水印准确率
                # =========================================================================
                print("\n7. 正在对 GT 相减后的图片进行水印准确率检测...")
                try:
                    subtracted_acc = evaluate_watermark_accuracy(subtracted_save_path)
                    print("=" * 70)
                    print(f"   [GT - 残差] 图片的水印提取准确率: {subtracted_acc:.4f}")
                    print("=" * 70)
                except Exception as e:
                    print(f"   [检测失败] 无法在 GT 相减图片上提取水印，报错信息: {e}")

                # =========================================================================
                # 8. 【新增】用 test_path 减去 (D1-D2) 残差，并测试准确率
                # =========================================================================
                print("\n8. 正在用 test_path 图片减去 (D1-D2) 残差，生成测试图片...")
                if not os.path.exists(args.test_path):
                    print(f"【错误】未找到 test_path 图片，请检查路径: {args.test_path}")
                else:
                    # 读取 test 图片，并确保尺寸一致 (resize=True 会将其调整为 512x512)
                    image_test, _, _ = load_image(args.test_path, resize=True)
                    test_float = image_test.astype(np.float32)

                    # 相减操作
                    test_subtracted_np = test_float - residual_float
                    test_subtracted_clipped_np = np.clip(test_subtracted_np, 0, 255).astype(np.uint8)

                    # 保存生成的相减图片
                    test_subtracted_save_path = os.path.join(args.save_path, "04_test_minus_residual.png")
                    cv2.imwrite(test_subtracted_save_path, cv2.cvtColor(test_subtracted_clipped_np, cv2.COLOR_RGB2BGR))
                    print(f"--> 已将 test_path 相减后的图片保存至: {test_subtracted_save_path}")

                    # 测试 test 相减后图片的水印准确率
                    print("\n正在对 test_path 相减后的图片进行水印准确率检测...")
                    try:
                        test_subtracted_acc = evaluate_watermark_accuracy(test_subtracted_save_path)
                        print("=" * 70)
                        print(f"   [Test - 残差] 图片的水印提取准确率: {test_subtracted_acc:.4f}")
                        print("=" * 70)
                    except Exception as e:
                        print(f"   [检测失败] 无法在 test_path 相减图片上提取水印，报错信息: {e}")

            print("流程结束。")

        # =========================================================================
        # 8. 【新增】用 test_path 减去 (D1-D2) 残差，并测试准确率
        # =========================================================================
        print("\n8. 正在用 test_path 图片减去 (D1-D2) 残差，生成测试图片...")
        if not os.path.exists(args.test_path):
            print(f"【错误】未找到 test_path 图片，请检查路径: {args.test_path}")
        else:
            # 读取 test 图片，并确保尺寸一致 (resize=True 会将其调整为 512x512)
            image_test, _, _ = load_image(args.test_path, resize=True)
            test_float = image_test.astype(np.float32)

            # 相减操作
            test_subtracted_np = test_float - residual_float
            test_subtracted_clipped_np = np.clip(test_subtracted_np, 0, 255).astype(np.uint8)

            # 保存生成的相减图片
            test_subtracted_save_path = os.path.join(args.save_path, "04_test_minus_residual.png")
            cv2.imwrite(test_subtracted_save_path, cv2.cvtColor(test_subtracted_clipped_np, cv2.COLOR_RGB2BGR))
            print(f"--> 已将 test_path 相减后的图片保存至: {test_subtracted_save_path}")

            # 测试 test 相减后图片的水印准确率
            print("\n正在对 test_path 相减后的图片进行水印准确率检测...")
            try:
                test_subtracted_acc = evaluate_watermark_accuracy(test_subtracted_save_path)
                print("=" * 70)
                print(f"   [Test - 残差] 图片的水印提取准确率: {test_subtracted_acc:.4f}")
                print("=" * 70)
            except Exception as e:
                print(f"   [检测失败] 无法在 test_path 相减图片上提取水印，报错信息: {e}")

    print("流程结束。")
