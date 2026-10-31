import os
import json
import csv
import glob
import cv2
import torch
import argparse
import numpy as np
import time
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)

try:
    from hidden_watermarker import HiDDeNWatermarker
    from basicsr.metrics import calculate_psnr, calculate_ssim
except ImportError as e:
    print(
        "警告: 未找到相关模块，"
        f"请确保文件在当前路径或 PYTHONPATH 中: {e}"
    )

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

inpaint_pipe = None

CSV_FIELDS = [
    "sample_id", "x_path", "p_x", "p_y", "seed",
    "num_steps", "guidance_scale", "beta", "eta", "tau_s", "t_s",
    "y_path", "yw_path", "watermark_scheme", "method", "xw_path",
    "semantic_distance", "ref_y_bitacc", "ref_yw_bitacc",
    "bitacc", "psnr", "ssim", "lpips", "clip_px", "clip_py",
]

SUMMARY_FIELDS = [
    "sample_id", "semantic_distance", "latent_ratio",
    "bitacc_direct", "bitacc_single", "bitacc_inpainting", "bitacc_ours",
    "psnr_direct", "psnr_single", "psnr_inpainting", "psnr_ours",
    "ssim_direct", "ssim_single", "ssim_inpainting", "ssim_ours",
    "lpips_direct", "lpips_single", "lpips_inpainting", "lpips_ours",
    "clip_px_direct", "clip_px_single", "clip_px_inpainting", "clip_px_ours",
    "clip_py_direct", "clip_py_single", "clip_py_inpainting", "clip_py_ours",
    "clip_margin_direct", "clip_margin_single", "clip_margin_inpainting", "clip_margin_ours",
]


def load_image(image_path, resize=False):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if resize:
        image = image.resize((512, 512))
    else:
        width_padding = (-w) % 8
        height_padding = (-h) % 8
        image = ImageOps.expand(image, (0, 0, width_padding, height_padding), fill=None)
    return np.array(image), h, w


def l2_norm(x):
    x = x.float()
    return x.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)


class ConfigurationOptimizer:
    def __init__(self, tau_s=0.5, eps=1e-4):
        self.tau_s = tau_s
        self.eps = eps

    def compute_t_s(self, scheduler):
        alphas_cumprod = scheduler.alphas_cumprod
        qualifying = []
        for t in scheduler.timesteps:
            t_int = int(t)
            alpha = alphas_cumprod[t_int].float()
            snr = alpha / (1.0 - alpha).clamp(min=self.eps)
            if float(snr) >= self.tau_s:
                qualifying.append(t_int)
        if qualifying:
            return int(max(qualifying))
        return 0


class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50, guidance_scale=1.0):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.guidance_scale = guidance_scale
        if self.model:
            self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.context_x = None
        self.context_y = None
        self.eps = 1e-4
        self.t_s = 0
        self.beta = 0.9
        self.eta = 0.01

    def ddim_reverse_step(self, model_output, current_t, prev_t, sample):
        sample = sample.float()
        model_output = model_output.float()
        current_t = int(current_t)

        alpha_t = self.model.scheduler.alphas_cumprod[current_t].to(sample.device).float()
        if prev_t is None:
            alpha_prev = self.model.scheduler.final_alpha_cumprod.to(sample.device).float()
        else:
            alpha_prev = self.model.scheduler.alphas_cumprod[int(prev_t)].to(sample.device).float()

        beta_t = 1.0 - alpha_t
        pred_x0 = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        prev_sample = alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * model_output
        return prev_sample

    def ddim_inversion_step(self, model_output, current_t, next_t, sample):
        sample = sample.float()
        model_output = model_output.float()
        if current_t is None:
            alpha_current = self.model.scheduler.final_alpha_cumprod.to(sample.device).float()
        else:
            alpha_current = self.model.scheduler.alphas_cumprod[int(current_t)].to(sample.device).float()

        alpha_next = self.model.scheduler.alphas_cumprod[int(next_t)].to(sample.device).float()
        beta_current = 1.0 - alpha_current
        pred_x0 = (sample - beta_current.sqrt() * model_output) / alpha_current.sqrt()
        next_sample = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * model_output
        return next_sample

    @torch.no_grad()
    def predict_eps(self, latents, t, context):
        latents = latents.half()
        context = context.half()
        uncond_embeddings, cond_embeddings = context.chunk(2)

        if self.guidance_scale == 1.0:
            return self.model.unet(latents, t, cond_embeddings).sample

        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings).sample
        noise_pred_text = self.model.unet(latents, t, cond_embeddings).sample
        return noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

    @torch.no_grad()
    def invert_trajectory(self, latent, context):
        z = latent.clone().float()
        timesteps = [int(t) for t in self.model.scheduler.timesteps]
        inversion_timesteps = list(reversed(timesteps))
        traj = [z.clone()]
        current_t = None

        for next_t in inversion_timesteps:
            model_t = 0 if current_t is None else current_t
            noise_pred = self.predict_eps(z.half(), model_t, context).float()
            z = self.ddim_inversion_step(noise_pred, current_t=current_t, next_t=next_t, sample=z)
            traj.append(z.clone())
            current_t = next_t
        return z, traj

    @torch.no_grad()
    def reverse_from(self, latent, context, num_steps=None):
        z = latent.clone().float()
        timesteps = [int(t) for t in self.model.scheduler.timesteps]
        if num_steps is None:
            active_timesteps = timesteps
        else:
            active_timesteps = timesteps[-int(num_steps):]

        n = len(active_timesteps)
        for i in range(n):
            current_t = active_timesteps[i]
            prev_t = active_timesteps[i + 1] if i + 1 < n else None
            noise_pred = self.predict_eps(z.half(), current_t, context).float()
            z = self.ddim_reverse_step(noise_pred, current_t=current_t, prev_t=prev_t, sample=z)
        return z

    def calibrate_residual(self, v, s, delta_ref, z_y):
        v, s, delta_ref, z_y = v.float(), s.float(), delta_ref.float(), z_y.float()
        R_t = l2_norm(delta_ref) / (l2_norm(z_y) + self.eps)
        scale = R_t * l2_norm(s) / (l2_norm(v) + self.eps)
        return v * scale

    @torch.no_grad()
    def jrt_step(self, s, r, current_t, prev_t, delta_ref_next, z_y_next):
        s_half, r_f = s.half(), r.float()

        eps_s = self.predict_eps(s_half, current_t, self.context_x).float()
        s_next = self.ddim_reverse_step(eps_s, current_t=current_t, prev_t=prev_t, sample=s_half).float()

        perturbed = s_half + self.eta * r_f.half()
        eps_sw = self.predict_eps(perturbed, current_t, self.context_x).float()
        s_w_next = self.ddim_reverse_step(eps_sw, current_t=current_t, prev_t=prev_t, sample=perturbed).float()

        r_tilde = (s_w_next - s_next) / self.eta
        r_cal = self.calibrate_residual(r_tilde, s_next, delta_ref_next, z_y_next).float()
        r_mix = self.beta * r_f + (1.0 - self.beta) * r_cal
        r_next = self.calibrate_residual(r_mix, s_next, delta_ref_next, z_y_next).float()
        r_next = torch.nan_to_num(r_next, nan=0.0, posinf=0.0, neginf=0.0)
        return s_next, r_next

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
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            return image
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        image = image.to(torch.float16)
        return (self.model.vae.encode(image)['latent_dist'].mean * 0.18215).to(torch.float16)

    @torch.no_grad()
    def init_prompt(self, p_x: str, p_y: str):
        def encode_text(p):
            uncond_input = self.model.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            text_input = self.model.tokenizer([p], padding="max_length", max_length=77, truncation=True,
                                              return_tensors="pt")
            return torch.cat([
                self.model.text_encoder(uncond_input.input_ids.to(device))[0],
                self.model.text_encoder(text_input.input_ids.to(device))[0]
            ]).to(torch.float16)

        self.context_x = encode_text(p_x)
        self.context_y = encode_text(p_y)


@torch.no_grad()
def run_jrt_transport(ode, traj_x, traj_y, delta_ref_traj):
    s_x = traj_x[-1].float()
    r_x = None
    timesteps = [int(t) for t in ode.model.scheduler.timesteps]
    T = len(timesteps)

    for t_idx in range(T):
        current_t = timesteps[t_idx]
        prev_t = timesteps[t_idx + 1] if t_idx + 1 < T else None
        k = T - t_idx

        if current_t > ode.t_s:
            eps_s = ode.predict_eps(s_x.half(), current_t, ode.context_x).float()
            s_x = ode.ddim_reverse_step(eps_s, current_t=current_t, prev_t=prev_t, sample=s_x).float()
            continue

        if r_x is None:
            r_x = ode.calibrate_residual(delta_ref_traj[k], s_x, delta_ref_traj[k], traj_y[k]).float()

        s_x, r_x = ode.jrt_step(s_x, r_x, current_t=current_t, prev_t=prev_t,
                                delta_ref_next=delta_ref_traj[k - 1], z_y_next=traj_y[k - 1])
    return s_x, r_x


@torch.no_grad()
def diffusion_inpainting(image_yw_np, p_x, seed, num_steps, guidance_scale, strength):
    if inpaint_pipe is None:
        raise RuntimeError("Diffusion Inpainting pipeline is not loaded.")
    init_image = Image.fromarray(image_yw_np).convert("RGB")
    mask_image = Image.new("L", init_image.size, 255)
    generator = torch.Generator(device=device).manual_seed(seed)
    out = inpaint_pipe(
        prompt=p_x, image=init_image, mask_image=mask_image, strength=strength,
        num_inference_steps=num_steps, guidance_scale=guidance_scale, generator=generator,
    ).images[0]
    return np.array(out.convert("RGB").resize((512, 512)))


def save_rgb(path, image_np):
    cv2.imwrite(path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


class MetricModels:
    def __init__(self, watermarker=None):
        self.watermarker = watermarker
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None
        self._load_lpips()
        self._load_clip()

    def _load_lpips(self):
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device).eval()
            print("--> LPIPS 已加载一次")
        except Exception as e:
            print(f"LPIPS 不可用: {e}")

    def _load_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(
                device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("--> CLIP 已加载一次 (safetensors)")
        except Exception as e:
            print(f"CLIP 不可用: {e}")
            self.clip_model = None
            self.clip_processor = None

    def bitacc(self, path):
        if self.watermarker is None:
            return float("nan")
        try:
            return float(self.watermarker.evaluate_accuracy(path))
        except Exception as e:
            print(f"HiDDeN BitAcc 失败: {e}")
            return float("nan")

    def lpips(self, x_np, xw_np):
        if self.lpips_model is None:
            return float('nan')

        def to_t(im):
            t = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            return t.to(device)

        with torch.no_grad():
            return float(self.lpips_model(to_t(x_np), to_t(xw_np)).item())

    def clip_scores(self, xw_np, p_x, p_y):
        if self.clip_model is None or self.clip_processor is None:
            return float('nan'), float('nan')
        image = Image.fromarray(xw_np)
        inputs = self.clip_processor(text=[p_x, p_y], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.clip_model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims = (img @ txt.T).squeeze(0)
        return float(sims[0].item()), float(sims[1].item())

    def clip_image_distance(self, img1_np, img2_np):
        if self.clip_model is None or self.clip_processor is None:
            return float('nan')
        images = [Image.fromarray(img1_np), Image.fromarray(img2_np)]
        inputs = self.clip_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            feats = self.clip_model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = (feats[0] * feats[1]).sum()
        return float(1.0 - sim.item())


def evaluate_xw(x_np, xw_np, xw_path, p_x, p_y, metrics):
    try:
        psnr = float(calculate_psnr(x_np, xw_np, crop_border=0, input_order='HWC', test_y_channel=False))
    except Exception as e:
        print(f"PSNR 失败: {e}")
        psnr = float('nan')
    try:
        ssim = float(calculate_ssim(x_np, xw_np, crop_border=0, input_order='HWC', test_y_channel=False))
    except Exception as e:
        print(f"SSIM 失败: {e}")
        ssim = float('nan')
    clip_px, clip_py = metrics.clip_scores(xw_np, p_x, p_y)
    return {
        "bitacc": metrics.bitacc(xw_path), "psnr": psnr, "ssim": ssim,
        "lpips": metrics.lpips(x_np, xw_np), "clip_px": clip_px, "clip_py": clip_py,
    }


def format_metrics(name, m):
    return (f"[{name}] BitAcc={m['bitacc']:.4f}  PSNR={m['psnr']:.4f}  "
            f"SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}  "
            f"CLIP-pX={m['clip_px']:.4f}  CLIP-pY={m['clip_py']:.4f}")


def discover_coco(coco_root, caption_json=None):
    if caption_json and os.path.isfile(caption_json):
        json_path = caption_json
    else:
        json_candidates = [
            os.path.join(coco_root, "annotations", "captions_val2017.json"),
            os.path.join(coco_root, "captions_val2017.json"),
            os.path.join(coco_root, "captions.json"),
        ]
        json_candidates += glob.glob(os.path.join(coco_root, "**", "*caption*.json"), recursive=True)
        json_path = next((p for p in json_candidates if os.path.isfile(p)), None)

    img_dirs = [
        os.path.join(coco_root, "val2017"),
        os.path.join(coco_root, "images", "val2017"),
        os.path.join(coco_root, "images"),
        coco_root,
    ]
    image_dir = None
    for d in img_dirs:
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.jpg")):
            image_dir = d
            break
    return json_path, image_dir


def load_coco_pairs(coco_root, num_samples, seed, caption_json=None):
    json_path, image_dir = discover_coco(coco_root, caption_json)
    if image_dir is None:
        raise FileNotFoundError(f"在 {coco_root} 下找不到 jpg")
    if json_path is None:
        raise FileNotFoundError(f"在 {coco_root} 下找不到 caption json，请加 --caption_json")
    print(f"--> COCO images: {image_dir}")
    print(f"--> COCO captions: {json_path}")

    file_to_path = {}
    id_to_file = {}
    for p in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        name = os.path.basename(p)
        file_to_path[name] = p
        try:
            id_to_file[int(os.path.splitext(name)[0])] = name
        except ValueError:
            pass

    with open(json_path, "r") as f:
        data = json.load(f)
    images = {im["id"]: im["file_name"] for im in data.get("images", [])}
    captions = {}
    for ann in data.get("annotations", []):
        fname = images.get(ann["image_id"], id_to_file.get(ann["image_id"]))
        if fname and fname not in captions:
            captions[fname] = ann["caption"].strip()

    pairs = []
    for fname, path in file_to_path.items():
        cap = captions.get(fname)
        if cap:
            pairs.append({"file_name": fname, "path": path, "caption": cap})
    if not pairs:
        raise RuntimeError("图片和 caption 对不上，检查文件名是否为 000000xxxxxx.jpg")

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)
    selected = [pairs[int(i)] for i in idx]
    n_all = len(pairs)
    for i, item in enumerate(selected):
        other = pairs[(int(idx[i]) + n_all // 2) % n_all]
        p_y = other["caption"]
        if p_y.strip().lower() == item["caption"].strip().lower():
            p_y = pairs[(int(idx[i]) + 1) % n_all]["caption"]
        item["p_x"] = item["caption"]
        item["p_y"] = p_y
        item["sample_id"] = os.path.splitext(item["file_name"])[0]
    return selected


def append_csv(csv_path, rows):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_summary_csv(csv_path, rows):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finished_sample_ids(csv_path, requested_method="all"):
    if not os.path.isfile(csv_path):
        return set()
    methods_by_sample = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            methods_by_sample.setdefault(sid, set()).add(row["method"])

    if requested_method == "all":
        required = {"direct", "single", "inpainting", "ours"}
    elif requested_method == "jrt":
        required = {"ours"}
    else:
        required = {requested_method}

    return {sid for sid, methods in methods_by_sample.items() if required.issubset(methods)}


def run_one_sample(args, ode, watermarker, metrics, sample_dir, gt_path, p_x, p_y, sample_id):
    os.makedirs(sample_dir, exist_ok=True)
    ode.init_prompt(p_x, p_y)
    print(f"p_X: {p_x}")
    print(f"p_Y: {p_y}")

    image_gt, _, _ = load_image(gt_path, resize=True)
    x_path = os.path.join(sample_dir, "02_x.png")
    save_rgb(x_path, image_gt)
    image_gt_latent = ode.image2latent(image_gt)

    print("1. Prepare clean decoy Y")
    hidden_path = os.path.join(sample_dir, "03_y.png")
    x_T, traj_x = ode.invert_trajectory(image_gt_latent, ode.context_x)

    if os.path.exists(hidden_path):
        print("--> reuse existing clean decoy Y")
        image_hidden_np, _, _ = load_image(hidden_path, resize=True)
    else:
        print("--> generate clean decoy Y")
        y_latent = ode.reverse_from(x_T, ode.context_y)
        image_hidden_np = ode.latent2image(y_latent)
        save_rgb(hidden_path, image_hidden_np)

    print("2. HiDDeN encoder 只调用一次 -> Y_w")
    yw_path = os.path.join(sample_dir, "03_yw.png")

    if os.path.exists(yw_path):
        print("--> reuse existing Yw; no new encoder query")
        image_hide_wm_np, _, _ = load_image(yw_path, resize=True)
    else:
        print("--> one HiDDeN encoder query")
        tmp1 = os.path.join(sample_dir, "03_y_encoded.png")
        tmp2 = os.path.join(sample_dir, "03_hidden_image_encoded.png")
        for p in [tmp1, tmp2]:
            if os.path.exists(p):
                os.remove(p)

        watermarker.add_watermark(hidden_path, save_dir=sample_dir, output_size=(512, 512))

        if os.path.exists(tmp1):
            encoded_path = tmp1
        elif os.path.exists(tmp2):
            encoded_path = tmp2
        else:
            raise RuntimeError("HiDDeN encoder did not produce output.")

        image_hide_wm_np, _, _ = load_image(encoded_path, resize=True)
        save_rgb(yw_path, image_hide_wm_np)

    image_y_latent = ode.image2latent(image_hidden_np)
    image_yw_latent = ode.image2latent(image_hide_wm_np)

    run_direct = args.method in ('all', 'direct')
    run_single = args.method in ('all', 'single')
    run_inpaint = args.method in ('all', 'inpainting')
    run_jrt = args.method in ('all', 'jrt')

    generated_outputs = {}
    xw_paths = {}

    if run_direct:
        print("3a. Direct Editing: invert Y_w with p_X, reverse with p_X")
        yw_T_x, _ = ode.invert_trajectory(image_yw_latent, ode.context_x)
        xw_direct = ode.latent2image(ode.reverse_from(yw_T_x, ode.context_x))
        xw_paths['direct'] = os.path.join(sample_dir, "xw_direct.png")
        save_rgb(xw_paths['direct'], xw_direct)
        generated_outputs['direct'] = xw_direct

    if run_single:
        print("3b. Single-Branch: invert Y_w with p_Y, reverse with p_X")
        yw_T_y, _ = ode.invert_trajectory(image_yw_latent, ode.context_y)
        xw_single = ode.latent2image(ode.reverse_from(yw_T_y, ode.context_x))
        xw_paths['single'] = os.path.join(sample_dir, "xw_single.png")
        save_rgb(xw_paths['single'], xw_single)
        generated_outputs['single'] = xw_single

    if run_inpaint:
        print("3c. Diffusion Inpainting")
        xw_inpaint = diffusion_inpainting(
            image_hide_wm_np, p_x, args.seed, args.num_steps, args.guidance_scale, args.inpaint_strength,
        )
        xw_paths["inpainting"] = os.path.join(sample_dir, "xw_inpainting.png")
        save_rgb(xw_paths["inpainting"], xw_inpaint)
        generated_outputs["inpainting"] = xw_inpaint

    if run_jrt:
        print("3d. Ours: paired reference trajectories + Jacobian-Guided Watermark Residual Transport")
        _, traj_y = ode.invert_trajectory(image_y_latent, ode.context_y)
        _, traj_yw = ode.invert_trajectory(image_yw_latent, ode.context_y)

        assert len(traj_y) == ode.num_ddim_steps + 1
        assert len(traj_yw) == ode.num_ddim_steps + 1

        delta_ref_traj = [traj_yw[k] - traj_y[k] for k in range(len(traj_y))]

        config_opt = ConfigurationOptimizer(tau_s=args.tau_s)

        if args.ts is not None:
            ode.t_s = args.ts
            print(f"--> 手动指定 t_s = {ode.t_s} (diffusion timestep)")
        else:
            ode.t_s = config_opt.compute_t_s(ode.model.scheduler)
            print(f"--> SNR 估计 t_s = {ode.t_s} (tau_s={args.tau_s})")

        print(f"--> t_s={ode.t_s}, eta={ode.eta}, beta={ode.beta}")

        s_ours, r_ours = run_jrt_transport(ode, traj_x, traj_y, delta_ref_traj)

        s_norm = float(l2_norm(s_ours.float()).mean().item())
        r_norm = float(l2_norm(r_ours.float()).mean().item())
        r_ratio = float((l2_norm(r_ours.float()) / (l2_norm(s_ours.float()) + ode.eps)).mean().item())

        print(f"[JRT diagnostic] ||s||={s_norm:.6f}  ||r||={r_norm:.6f}  ||r||/||s||={r_ratio:.6f}")

        xw_ours = ode.latent2image(s_ours + r_ours)
        xw_paths["ours"] = os.path.join(sample_dir, "xw_ours.png")
        save_rgb(xw_paths["ours"], xw_ours)
        generated_outputs["ours"] = xw_ours

    print("\n4. Offline evaluation only")
    clean_y_bitacc = metrics.bitacc(hidden_path)
    ref_bitacc = metrics.bitacc(yw_path)
    semantic_distance = metrics.clip_image_distance(image_gt, image_hidden_np)

    print(f"[Reference Y  BitAcc] {clean_y_bitacc:.4f}")
    print(f"[Reference Yw BitAcc] {ref_bitacc:.4f}")
    print(f"[semantic_distance(X,Y)] {semantic_distance:.4f}")

    try:
        ref_psnr = float(
            calculate_psnr(image_hidden_np, image_hide_wm_np, crop_border=0, input_order='HWC', test_y_channel=False))
    except Exception:
        ref_psnr = float('nan')
    try:
        ref_ssim = float(
            calculate_ssim(image_hidden_np, image_hide_wm_np, crop_border=0, input_order='HWC', test_y_channel=False))
    except Exception:
        ref_ssim = float('nan')
    ref_lpips = metrics.lpips(image_hidden_np, image_hide_wm_np)
    latent_r0 = float((l2_norm(image_yw_latent.float() - image_y_latent.float()) / (
                l2_norm(image_y_latent.float()) + ode.eps)).mean().item())
    print(
        f"[Y vs Yw] PSNR={ref_psnr:.4f} SSIM={ref_ssim:.4f} LPIPS={ref_lpips:.4f} | latent_ratio(||Yw-Y||/||Y||)={latent_r0:.4f}")

    all_metrics = {}
    for method, xw_np in generated_outputs.items():
        all_metrics[method] = evaluate_xw(image_gt, xw_np, xw_paths[method], p_x, p_y, metrics)
        print(format_metrics(method, all_metrics[method]))

    def _m(method, field="bitacc"):
        d = all_metrics.get(method)
        if d is None:
            return float("nan")
        return float(d[field])

    def _margin(method):
        d = all_metrics.get(method)
        if d is None:
            return float("nan")
        return float(d["clip_px"] - d["clip_py"])

    summary_row = {
        "sample_id": sample_id, "semantic_distance": semantic_distance, "latent_ratio": latent_r0,
        "bitacc_direct": _m('direct'), "bitacc_single": _m('single'), "bitacc_inpainting": _m('inpainting'),
        "bitacc_ours": _m('ours'),
        "psnr_direct": _m('direct', 'psnr'), "psnr_single": _m('single', 'psnr'),
        "psnr_inpainting": _m('inpainting', 'psnr'), "psnr_ours": _m('ours', 'psnr'),
        "ssim_direct": _m('direct', 'ssim'), "ssim_single": _m('single', 'ssim'),
        "ssim_inpainting": _m('inpainting', 'ssim'), "ssim_ours": _m('ours', 'ssim'),
        "lpips_direct": _m('direct', 'lpips'), "lpips_single": _m('single', 'lpips'),
        "lpips_inpainting": _m('inpainting', 'lpips'), "lpips_ours": _m('ours', 'lpips'),
        "clip_px_direct": _m('direct', 'clip_px'), "clip_px_single": _m('single', 'clip_px'),
        "clip_px_inpainting": _m('inpainting', 'clip_px'), "clip_px_ours": _m('ours', 'clip_px'),
        "clip_py_direct": _m('direct', 'clip_py'), "clip_py_single": _m('single', 'clip_py'),
        "clip_py_inpainting": _m('inpainting', 'clip_py'), "clip_py_ours": _m('ours', 'clip_py'),
        "clip_margin_direct": _margin('direct'), "clip_margin_single": _margin('single'),
        "clip_margin_inpainting": _margin('inpainting'), "clip_margin_ours": _margin('ours'),
    }

    results_path = os.path.join(sample_dir, f"results_{args.method}.txt")
    with open(results_path, "w") as f:
        f.write(f"p_x: {p_x}\n p_y: {p_y}\n")
        f.write(f"Reference Y BitAcc: {clean_y_bitacc:.4f}\n")
        f.write(f"Reference Yw BitAcc: {ref_bitacc:.4f}\n")
        f.write(f"semantic_distance: {semantic_distance:.4f}\n")
        f.write(
            f"Y_vs_Yw: psnr={ref_psnr:.4f}, ssim={ref_ssim:.4f}, lpips={ref_lpips:.4f}, latent_ratio={latent_r0:.4f}\n")
        for name, m in all_metrics.items():
            f.write(format_metrics(name, m) + "\n")

    rows = []
    for method, m in all_metrics.items():
        rows.append({
            "sample_id": sample_id, "x_path": x_path, "p_x": p_x, "p_y": p_y, "seed": args.seed,
            "num_steps": args.num_steps, "guidance_scale": args.guidance_scale, "beta": args.beta, "eta": args.eta,
            "tau_s": args.tau_s,
            "t_s": ode.t_s if method == "ours" else "",
            "y_path": hidden_path, "yw_path": yw_path, "watermark_scheme": "HiDDeN", "method": method,
            "xw_path": xw_paths.get(method, ""),
            "semantic_distance": semantic_distance, "ref_y_bitacc": clean_y_bitacc, "ref_yw_bitacc": ref_bitacc,
            "bitacc": m["bitacc"], "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"], "clip_px": m["clip_px"],
            "clip_py": m["clip_py"],
        })

    return rows, summary_row


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/coco_val')
    parser.add_argument('--caption_json', type=str, default=None, help='captions_val2017.json 的完整路径')

    # ==========================================
    # 重点修改 1：替换为官方 .pyt 权重路径
    # ==========================================
    parser.add_argument(
        '--hidden_weights_path',
        type=str,
        default='./combined-noise-repro--epoch-400.pyt'
    )

    # ==========================================
    # 重点修改 2：改为 30 bit
    # ==========================================
    parser.add_argument('--hidden_num_bits', type=int, default=30)

    # ==========================================
    # 重点修改 3：指定官方模型训练时的 H/W 尺寸
    # ==========================================
    parser.add_argument('--hidden_image_size', type=int, default=128)

    parser.add_argument('--save_path', type=str,
                        default='/data/opt_data_backup/data/jiang/cross/CRoSS/coco_val/hidden/hidden_vb_final')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--tau_s', type=float, default=0.5)
    parser.add_argument('--ts', type=int, default=None)
    parser.add_argument('--method', type=str, default='all', choices=['all', 'jrt', 'direct', 'single', 'inpainting'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_id', type=str, default=None, help='只跑指定 sample id（例如 000000563349）')
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument('--inpaint_model_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/coco_val/sd-inpainting')
    parser.add_argument('--inpaint_strength', type=float, default=1.0)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ode = ODESolve(ldm_stable, args.num_steps, guidance_scale=args.guidance_scale)
    ode.beta = args.beta
    ode.eta = args.eta

    if args.method in ('all', 'inpainting'):
        try:
            inpaint_scheduler = DDIMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                clip_sample=False, set_alpha_to_one=False, steps_offset=1,
            )
            inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.inpaint_model_path,
                torch_dtype=torch.float16,
                variant="fp16",  # 新增：用 fp16 变体权重
                scheduler=inpaint_scheduler,
                safety_checker=None,  # 新增：跳过 safety_checker
                requires_safety_checker=False,  # 新增
            ).to(device)
            print("--> Diffusion Inpainting pipeline loaded (DDIM scheduler)")
        except Exception as e:
            print(f"Diffusion Inpainting 模型加载失败: {e}")
            inpaint_pipe = None

    # ==========================================
    # 重点修改 4：根据 30 bit 重新生成密钥字符串（30个 0/1）
    # ==========================================
    HIDDEN_KEY_STR = (
        "111010110101000001010111010011"
    )

    hidden_key = [int(c) for c in HIDDEN_KEY_STR]

    assert len(hidden_key) == 30

    # ==========================================
    # 重点修改 5：用新的 weights_path 参数实例化封装类
    # ==========================================
    watermarker = HiDDeNWatermarker(
        weights_path=args.hidden_weights_path,
        device=device,
        num_bits=args.hidden_num_bits,
        image_size=args.hidden_image_size,
        key=hidden_key,
    )

    metrics = MetricModels(watermarker=watermarker)

    csv_path = os.path.join(args.save_path, "pilot.csv")
    manifest_path = os.path.join(args.save_path, f"coco_val2017_pairs_seed{args.seed}_n{args.num_samples}.json")

    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        pairs = []
        for item in manifest:
            pairs.append({
                "file_name": os.path.basename(item["x_path"]), "path": item["x_path"],
                "caption": item["p_x"], "p_x": item["p_x"], "p_y": item["p_y"], "sample_id": item["sample_id"],
            })
        print(f"--> loaded pairing manifest: {manifest_path}")
    else:
        pairs = load_coco_pairs(args.coco_root, args.num_samples, args.seed, args.caption_json)
        manifest = [{"sample_id": p["sample_id"], "x_path": p["path"], "p_x": p["p_x"], "p_y": p["p_y"]} for p in pairs]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"--> saved pairing manifest: {manifest_path}")

    done = finished_sample_ids(csv_path, args.method)

    if args.sample_id is not None:
        pairs = [p for p in pairs if p["sample_id"] == args.sample_id]
        if not pairs:
            raise RuntimeError(f"找不到 sample_id={args.sample_id}。")
        print(f"--> 仅运行 sample_id={args.sample_id}")

    print(f"--> 计划 {len(pairs)} 张, 已完成 {len(done)}")
    print(f"--> save_path: {args.save_path}")

    for item in pairs:
        sid = item["sample_id"]
        if sid in done:
            print(f"skip {sid}")
            continue
        print("\n" + "=" * 60)
        print(f"sample {sid}")
        rows, summary_row = run_one_sample(
            args, ode, watermarker, metrics,
            os.path.join(args.save_path, sid),
            item["path"], item["p_x"], item["p_y"], sid
        )
        append_csv(csv_path, rows)

    print("\n" + "=" * 60)
    print(f"[Total Time] {time.time() - start_time:.2f} s")
    print(f"CSV: {csv_path}")