import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import os
import pandas as pd
from typing import List
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ================= 配置与显卡设置 =================

DEVICE_ID = 'cuda:1'
device = torch.device(DEVICE_ID) if torch.cuda.is_available() else torch.device('cpu')

# 本地模型路径
MODEL_PATH = "/home/liyan/.cache/huggingface/stable-diffusion-v1-5"
MY_TOKEN = ''


# ================= 1. 定义 LambdaNet (权重预测网络) =================

class LambdaNet(nn.Module):
    """
    输入图像的 Latent (4x64x64)，输出空间 Lambda Map (1x64x64)
    用于动态调整图像不同区域的水印嵌入强度。
    """

    def __init__(self):
        super(LambdaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 限制输出在 0~1 之间
        )

    def forward(self, x):
        return self.net(x)


# ================= 2. 数据集加载类 (适配 output_train 结构) =================

class StegoDataset(Dataset):
    def __init__(self, excel_path, data_root, size=(512, 512)):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.size = size

    def __len__(self):
        return len(self.df)

    def load_image_as_tensor(self, path):
        if not os.path.exists(path): return None
        image = Image.open(path).convert("RGB").resize(self.size)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0  # 归一化到 [-1, 1]
        return torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_name = str(row['Number'])
        p1 = str(row['prompt1'])
        p2 = str(row['prompt2'])

        folder_path = os.path.join(self.data_root, folder_name)
        # 处理文件夹后缀匹配 (针对 .jpg 或 .png)
        if not os.path.exists(folder_path):
            if os.path.exists(folder_path + ".jpg"):
                folder_path += ".jpg"
            elif os.path.exists(folder_path + ".png"):
                folder_path += ".png"
            else:
                return None

        gt_tensor = self.load_image_as_tensor(os.path.join(folder_path, "gt.png"))
        encoded_tensor = self.load_image_as_tensor(os.path.join(folder_path, "hide_encoded.png"))

        if gt_tensor is None or encoded_tensor is None: return None
        return gt_tensor, encoded_tensor, p1, p2


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)


# ================= 3. ODESolve (修复 Scheduler 报错并支持梯度) =================

class ODESolve:
    def __init__(self, model_path, num_ddim_steps=20):
        # 修复逻辑：手动初始化调度器以避免缺失 scheduler_config.json 的错误
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )

        # 加载 Pipeline，显示传入手动配置的 scheduler
        self.model = StableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=self.scheduler,
            local_files_only=True
        ).to(device)

        self.num_ddim_steps = num_ddim_steps
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        # 冻结基础模型参数，确保只训练 LambdaNet
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(False)
        self.image_enhide_latents = None

    def step_custom(self, model_output, timestep, sample, is_forward=True):
        """通用的 DDIM 步进逻辑，支持梯度传播"""
        step_size = 1000 // self.num_ddim_steps
        t = timestep.long()

        # 计算 next_t
        if is_forward:
            next_t = min(t + step_size, torch.tensor(999).to(device))
        else:
            next_t = t - step_size

        alpha_prod_t = self.model.scheduler.alphas_cumprod[t] if t >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_next = self.model.scheduler.alphas_cumprod[
            next_t] if next_t >= 0 else self.model.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        sample_direction = (1 - alpha_prod_next) ** 0.5 * model_output
        return alpha_prod_next ** 0.5 * pred_original_sample + sample_direction

    def get_noise_pred(self, latents, t, context, lamda_map=None):
        latents = latents.to(self.model.unet.dtype)
        uncond_embeddings, cond_embeddings = context.chunk(2)

        # 三路预测：无条件、文本引导、水印引导
        noise_uncond = self.model.unet(latents, t, encoder_hidden_states=uncond_embeddings)["sample"]
        noise_text = self.model.unet(latents, t, encoder_hidden_states=cond_embeddings)["sample"]
        noise_wm = self.model.unet(self.image_enhide_latents, t, encoder_hidden_states=cond_embeddings)["sample"]

        # 空间自适应混合公式
        current_lambda = lamda_map if lamda_map is not None else 0.5
        noise_pred = noise_uncond + 1.0 * (noise_text - noise_uncond) + current_lambda * (noise_wm - noise_uncond)

        # 同步更新水印参考轨迹的预测值
        noise_en_pred = noise_uncond + 1.0 * (noise_wm - noise_uncond)
        return noise_pred, noise_en_pred

    def ddim_loop_train(self, latent, context, lamda_map, is_forward=True):
        curr_latent = latent.clone()
        timesteps = self.model.scheduler.timesteps

        for i in range(len(timesteps)):
            t = timesteps[len(timesteps) - i - 1] if is_forward else timesteps[i]
            noise_pred, noise_en_pred = self.get_noise_pred(curr_latent, t, context, lamda_map)

            # 更新当前 Latent 和 水印参考轨迹 Latent
            curr_latent = self.step_custom(noise_pred, t, curr_latent, is_forward=is_forward)
            self.image_enhide_latents = self.step_custom(noise_en_pred, t, self.image_enhide_latents,
                                                         is_forward=is_forward)

        return curr_latent

    @torch.no_grad()
    def get_batch_embeddings(self, prompts: List[str]):
        bs = len(prompts)
        uncond_input = self.model.tokenizer([""] * bs, padding="max_length", max_length=77, return_tensors="pt").to(
            device)
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids)[0]

        text_input = self.model.tokenizer(prompts, padding="max_length", max_length=77, truncation=True,
                                          return_tensors="pt").to(device)
        text_embeddings = self.model.text_encoder(text_input.input_ids)[0]

        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def image2latent(self, image_tensor):
        latents = self.model.vae.encode(image_tensor.to(device))['latent_dist'].mean
        return latents * 0.18215


# ================= 4. 训练主逻辑 =================

def train_framework(args):
    # 初始化 ODE Solver (包含 Pipeline 和手动配置的 Scheduler)
    ode = ODESolve(MODEL_PATH, num_ddim_steps=args.num_steps)

    ode.model.unet.enable_gradient_checkpointing()
    lambda_net = LambdaNet().to(device)
    optimizer = torch.optim.Adam(lambda_net.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    # 加载数据集
    dataset = StegoDataset(args.excel_path, args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"Starting training on {len(dataset)} images on device: {device}")

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if batch is None: continue
            clean_imgs, encoded_imgs, p1_list, p2_list = batch  # p1: private, p2: public

            # A. 准备阶段 (无梯度：计算 Latents 和 Embeddings)
            with torch.no_grad():
                latent_clean = ode.image2latent(clean_imgs).detach()
                latent_target = ode.image2latent(encoded_imgs).detach()

                context_public = ode.get_batch_embeddings(list(p2_list))
                context_private = ode.get_batch_embeddings(list(p1_list))

                # 前向加噪获取起始噪声点
                ode.image_enhide_latents = latent_target.clone()
                start_noise = ode.ddim_loop_train(latent_clean, context_public, lamda_map=None, is_forward=True)
                enhide_noise = ode.image_enhide_latents.clone()

            # B. 训练阶段 (有梯度：反向传播优化 LambdaNet)
            optimizer.zero_grad()

            # 基于原始图像 Latent 预测空间 Lambda Map
            pred_map = lambda_net(latent_clean)

            # 重置 ODE 状态用于反向重建
            ode.image_enhide_latents = enhide_noise.detach()

            # 执行带梯度的反向生成循环
            reconstructed_latent = ode.ddim_loop_train(
                start_noise.detach(),
                context_private,
                lamda_map=pred_map,
                is_forward=False
            )

            # C. 损失计算
            loss_quality = mse_loss(reconstructed_latent, latent_clean)
            loss_watermark = mse_loss(reconstructed_latent, latent_target)

            # 综合平衡损失：0.7 权重用于保真度，0.3 权重用于水印提取
            total_loss = 0.7 * loss_quality + 0.3 * loss_watermark

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(lambda_net.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "MapAvg": f"{pred_map.mean().item():.3f}"})

        # 定期保存权重
        if epoch % 5 == 0:
            torch.save(lambda_net.state_dict(), os.path.join(args.save_path, f'lambda_net_epoch_{epoch}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_path', type=str, default='/opt/data/jiang/cross/CRoSS/stega260.xlsx')
    parser.add_argument('--data_root', type=str, default='/opt/data/jiang/cross/CRoSS/output_train')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Model save directory')
    parser.add_argument('--num_steps', type=int, default=2, help='DDIM steps (low for VRAM efficiency)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (1 is recommended for VRAM)')
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    train_framework(args)