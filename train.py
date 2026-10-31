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

# 请确保当前目录下有这两个库，或者根据实际情况修改导入路径
try:
    from decode_image import evaluate_watermark_accuracy
    from basicsr.metrics import calculate_psnr, calculate_ssim
except ImportError:
    print("警告: 未找到 decode_image 或 basicsr 模块，请确保文件存在。")

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# 全局配置
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
MY_TOKEN = ''  # 如果需要 HuggingFace Token 请填入
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 加载模型
model_path_diffusion = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
# 如果本地没有模型，可以使用 "runwayml/stable-diffusion-v1-5" 并填入 Token
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


class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None
        # 初始化默认参数，防止未设置时报错
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

        # 计算各种条件的噪声预测
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]
        noise_prediction_wm = self.model.unet(self.image_enhide_latents, t, cond_embeddings)["sample"]

        # === 核心修改：使用两个优化参数 ===
        # 获取当前的优化参数 (如果是 Tensor 则取值，如果是数值则直接用)
        w_text = self.lamda_text
        w_wm = self.lamda_wm

        # 公式: noise = Uncond + w1 * (Text - Uncond) + w2 * (WM - Uncond)
        noise_pred = noise_pred_uncond + \
                     w_text * (noise_prediction_text - noise_pred_uncond) + \
                     w_wm * (noise_prediction_wm - noise_pred_uncond)

        # 隐写图像的演化逻辑 (保持原样，或根据需要修改)
        # 这里原来的逻辑是系数为 1，暂时保持不变
        noise_en_pred = noise_pred_uncond + 1 * (noise_prediction_wm - noise_pred_uncond)

        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
            self.image_enhide_latents = self.next_step(noise_en_pred, t, self.image_enhide_latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
            self.image_enhide_latents = self.prev_step(noise_en_pred, t, self.image_enhide_latents)
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
            if type(image) is Image.Image:  # 修正类型检查
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
        # 重置 image_enhide_latents 到初始状态 (对于反向过程很重要)
        # 注意：这里可能需要外部重新赋值初始 latents，或者在 invert 中处理

        for i in tqdm(range(self.num_ddim_steps), leave=False):  # 进度条设为 False 防止刷屏
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, prompt, start_latent, is_forward, en_latents_reset=None):
        self.init_prompt(prompt)
        # 如果提供了 en_latents_reset，则重置它（用于每次迭代开始时）
        if en_latents_reset is not None:
            self.image_enhide_latents = en_latents_reset.clone()

        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]


# === 定义二维搜索空间 ===
# lamda_text (val1): 范围 0.5 到 2.5
# lamda_wm (val2): 范围 0.0 到 1.0
space = [
    Real(0.5, 2.5, name='lamda_text'),
    Real(0.0, 1.0, name='lamda_wm')
]


def optimize_lamda_2d(ode: ODESolve,
                      latent_input,
                      enhide_latents_initial,  # 传入初始的加密图像 latent
                      image_gt,  # 显式传入 GT 图像
                      save_path: str,
                      prompt_private: str,
                      prompt_public: str,
                      n_calls: int = 40):  # 二维空间增加尝试次数
    """
    用贝叶斯优化调节 lamda_text 和 lamda_wm 两个参数
    """

    @use_named_args(space)
    def objective(**params):
        l_text = params['lamda_text']
        l_wm = params['lamda_wm']

        # 将参数传递给 ODE 对象 (转为 tensor 以保持一致性)
        ode.lamda_text = torch.tensor(l_text, dtype=torch.float32, device=device)
        ode.lamda_wm = torch.tensor(l_wm, dtype=torch.float32, device=device)

        # 1. 正向扩散 (Public Key)
        # 注意：每次开始都需要重置 image_enhide_latents 到初始状态
        latent_noise = ode.invert(prompt_public, latent_input, is_forward=True,
                                  en_latents_reset=enhide_latents_initial)

        # 2. 反向生成 (Private Key)
        # 这里的 en_latents_reset 应该使用正向过程结束后的状态吗？
        # 通常逻辑是：正向过程改变了 image_enhide_latents，反向过程接着用。
        # 或者是两个独立的过程？根据你之前的代码逻辑，这是一个连贯的过程。
        # 这里假设是连贯的，不需要在反向开始时重置，因为正向过程已经修改了 self.image_enhide_latents

        image_reverse_latent = ode.invert(prompt_private, latent_noise, is_forward=False)
        image_reverse = ode.latent2image(image_reverse_latent)

        # 保存临时图像
        reverse_path = os.path.join(save_path, 'reverse_tmp.png')
        cv2.imwrite(reverse_path, cv2.cvtColor(image_reverse, cv2.COLOR_RGB2BGR))

        # 计算指标
        try:
            acc = evaluate_watermark_accuracy(reverse_path)
        except Exception as e:
            print(f"水印评估出错: {e}, 设为0")
            acc = 0

        psnr = calculate_psnr(image_gt, image_reverse, crop_border=0, input_order='HWC', test_y_channel=False)

        # 综合得分
        score = psnr + acc * 25

        print(f"[Text={l_text:.3f} | WM={l_wm:.3f}] PSNR={psnr:.4f}, Acc={acc:.4f}, Score={score:.4f}")

        # 返回负值用于最小化
        return -score

    print("开始贝叶斯优化 (2D)...")
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

    print("Optimization Finished.")
    print(f"Best Text Weight (Val1): {res.x[0]:.4f}")
    print(f"Best Watermark Weight (Val2): {res.x[1]:.4f}")
    print(f"Best Score: {-res.fun:.4f}")

    return res.x


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--gt_path', type=str,
                        default='/opt/data/jiang/cross/CRoSS/image_jiang/gt.png', help='原始图像')
    parser.add_argument('--hideenimage_path', type=str,
                        default='/opt/data/jiang/cross/CRoSS/image_jiang/hide.png', help='隐藏图像')
    parser.add_argument('--hideimage_path', type=str,
                        default='/opt/data/jiang/cross/CRoSS/image_jiang/hide_encoded.png', help='加密图像')

    # Prompt 配置
    parser.add_argument('--private_key', type=str, default='a cat', help='private key prompt')
    parser.add_argument('--public_key', type=str, default='a tiger', help='public key prompt')

    # 其他配置
    parser.add_argument('--save_path', type=str, default='/opt/data/jiang/cross/CRoSS/image_jiang/train',
                        help='保存路径')
    parser.add_argument('--num_steps', type=int, default=50, help='DDIM采样步数')
    parser.add_argument('--opt_calls', type=int, default=100, help='贝叶斯优化调用次数')  # 建议稍微调大
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 初始化 ODE Solver
    ode = ODESolve(ldm_stable, args.num_steps)

    # 读取图像
    print("正在读取图像...")
    image_gt, _, _ = load_image(args.gt_path, resize=True)
    image_hide, _, _ = load_image(args.hideimage_path, resize=True)  # 注意：这里是你代码里的 hide_encoded
    image_enhide, _, _ = load_image(args.hideenimage_path, resize=True)  # 这里是你代码里的 hide

    # 转为 Latent
    image_hide_latent_reveal = ode.image2latent(image_hide)  # 作为输入的 Latent
    image_enhide_latents_initial = ode.image2latent(image_enhide)  # 作为条件的 Latent (Base)

    # 运行二维贝叶斯优化
    best_params = optimize_lamda_2d(
        ode=ode,
        latent_input=image_hide_latent_reveal,
        enhide_latents_initial=image_enhide_latents_initial,
        image_gt=image_gt,
        save_path=args.save_path,
        prompt_private=args.private_key,
        prompt_public=args.public_key,
        n_calls=args.opt_calls
    )

    # 使用最佳参数生成最终结果
    best_text_w, best_wm_w = best_params
    print(f"\n正在使用最佳参数生成最终图像: Text={best_text_w:.4f}, WM={best_wm_w:.4f}")

    ode.lamda_text = torch.tensor(best_text_w, dtype=torch.float32, device=device)
    ode.lamda_wm = torch.tensor(best_wm_w, dtype=torch.float32, device=device)

    # 最终生成流程
    final_latent_noise = ode.invert(args.public_key, image_hide_latent_reveal, is_forward=True,
                                    en_latents_reset=image_enhide_latents_initial)
    final_reverse_latent = ode.invert(args.private_key, final_latent_noise, is_forward=False)
    final_image = ode.latent2image(final_reverse_latent)

    # 保存最终结果
    final_save_path = os.path.join(args.save_path, 'best_result.png')
    cv2.imwrite(final_save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    final_psnr = calculate_psnr(image_gt, final_image, crop_border=0, input_order='HWC', test_y_channel=False)
    print(f"最终结果已保存: {final_save_path} | PSNR: {final_psnr:.4f}")