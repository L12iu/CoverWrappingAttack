"""
给原图加 StegaStamp 水印（独立于 JRT）。
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/data/opt_data_backup/data/jiang/cross/CRoSS")
from encode_image import StegaStampWatermarker

# ===== 路径配置 =====
INPUT_PATH  = "/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gt.png"
OUTPUT_PATH = "/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gauss/14/encoded_gt.png"
MODEL_PATH  = "/data/opt_data_backup/data/140000.pth"
# ===================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

watermarker = StegaStampWatermarker(MODEL_PATH)

tmp_dir = "/tmp/stega_embed_tmp"
os.makedirs(tmp_dir, exist_ok=True)

watermarked_img, _ = watermarker.add_watermark(
    INPUT_PATH,
    save_dir=tmp_dir,
    output_size=(512, 512)
)

watermarked_img.save(OUTPUT_PATH)
print(f"已保存: {OUTPUT_PATH}")

from decode_image import evaluate_watermark_accuracy
acc = evaluate_watermark_accuracy(OUTPUT_PATH)
print(f"闭环验证 Bit Acc: {acc:.4f}")