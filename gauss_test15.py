import os
import cv2
import argparse
import numpy as np
from PIL import Image, ImageOps

# 尝试导入自定义模块
try:
    from encode_image import StegaStampWatermarker
    from decode_image import evaluate_watermark_accuracy
except ImportError:
    print("警告: 未找到相关模块（StegaStamp 或 evaluate），请确保文件在当前路径或 PYTHONPATH 中。")


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
# 主程序
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gt.png')
    parser.add_argument('--test_path', type=str,
                        default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gauss/14/08_pinjie.png')
    parser.add_argument('--stegastamp_model', type=str,
                        default='/data/opt_data_backup/data/140000.pth')
    parser.add_argument('--save_path', type=str, default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gauss/13')

    # 读取已通过扩散模型生成的 A 和 B 图片
    parser.add_argument('--image_a_path', type=str, default=None, help="图片A (02_hidden_image_A) 的路径")
    parser.add_argument('--image_b_path', type=str, default=None, help="图片B (02_extra_image_B) 的路径")
    parser.add_argument('--d2_path', type=str,
                        default='/data/opt_data_backup/data/jiang/cross/CRoSS/image_jiang/gauss/9/02_hidden_image_residual.png')

    # 权重和亮度参数
    parser.add_argument('--weight1', type=float, default=0.8, help="GT/Test 图像的基础权重 (0-1), 默认0.8以降低亮度")
    parser.add_argument('--weight2', type=float, default=0.5, help="残差 (D1-D2) 的基础权重")
    parser.add_argument('--brightness_factor', type=float, default=1.0, help="相减后进一步调整亮度的因子")

    # 新增参数：动态区域权重控制
    parser.add_argument('--blur_size', type=int, default=31, help="高斯模糊核大小，用于计算区域差异，必须是奇数")
    parser.add_argument('--diff_sensitivity', type=float, default=1.0,
                        help="差异敏感度放大因子，越大则细微差异也会导致残差比重急剧下降")

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    image_a_path = args.image_a_path if args.image_a_path else os.path.join(args.save_path, "02_hidden_image_A.png")
    image_b_path = args.image_b_path if args.image_b_path else os.path.join(args.save_path, "02_extra_image_B.png")
    d2_path = args.d2_path

    # 0. 检查所有前置图片是否存在
    for path, name in zip([args.gt_path, image_a_path, image_b_path, d2_path],
                          ['GT', 'Image A', 'Image B', 'Image D2']):
        if not os.path.exists(path):
            print(f"【严重错误】未找到 {name} 图片，请检查路径: {path}")
            exit(1)

    # 1. 读取 GT
    print("1. 正在读取 GT 图片...")
    image_gt, h, w = load_image(args.gt_path, resize=True)
    cv2.imwrite(os.path.join(args.save_path, "01_gt.png"), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    # 2. 读取图片 A 和 B
    print("2. 正在读取预生成的图片 A 和 B...")
    image_hidden_np, _, _ = load_image(image_a_path, resize=True)
    image_B_np, _, _ = load_image(image_b_path, resize=True)

    # =========================================================================
    # 3. 计算残差 D1 以及 D1-D2
    # =========================================================================
    print("3. 正在计算残差 D1 以及 D1-D2...")
    D1 = image_B_np.astype(np.float32) - image_hidden_np.astype(np.float32)
    image_D2_np, _, _ = load_image(d2_path, resize=True)
    D1_minus_D2 = D1 - image_D2_np.astype(np.float32)

    D1_minus_D2_visual = np.clip(np.abs(D1_minus_D2), 0, 255).astype(np.uint8)
    d1_minus_d2_save_path = os.path.join(args.save_path, "02_extra_D1_minus_D2.png")
    cv2.imwrite(d1_minus_d2_save_path, cv2.cvtColor(D1_minus_D2_visual, cv2.COLOR_RGB2BGR))

    # =========================================================================
    # 4. 用 GT 按比例减去 (D1-D2) 残差，生成测试图片
    # =========================================================================
    print(f"\n4. 正在用 GT (权重={args.weight1}) 按比例减去 (D1-D2) 残差 (权重={args.weight2})...")
    gt_float = image_gt.astype(np.float32)
    residual_float = D1_minus_D2.astype(np.float32)

    subtracted_np = (args.weight1 * gt_float) - (args.weight2 * residual_float)
    if args.brightness_factor != 1.0:
        subtracted_np *= args.brightness_factor

    subtracted_clipped_np = np.clip(subtracted_np, 0, 255).astype(np.uint8)
    subtracted_save_path = os.path.join(args.save_path, f"03_gt_minus_residual.png")
    cv2.imwrite(subtracted_save_path, cv2.cvtColor(subtracted_clipped_np, cv2.COLOR_RGB2BGR))

    # 5. 测试 GT 相减后图片的水印准确率 (略去打印细节)
    try:
        subtracted_acc = evaluate_watermark_accuracy(subtracted_save_path)
    except Exception as e:
        pass

        # =========================================================================
    # 6. 【核心修改】引入空间动态权重的 Test 相减逻辑
    # =========================================================================
    print(f"\n6. 正在计算区域动态权重，并用 test_path 图片减去残差...")
    if not os.path.exists(args.test_path):
        print(f"【错误】未找到 test_path 图片，请检查路径: {args.test_path}")
    else:
        image_test, _, _ = load_image(args.test_path, resize=True)
        test_float = image_test.astype(np.float32)
        generated_float = image_B_np.astype(np.float32)  # 以 Image B 作为生成图像基准

        # 6.1 计算 Test 和生成图像的像素绝对差异
        diff_map = np.abs(test_float - generated_float)

        # 6.2 将三通道差异平均为单通道，便于评估整体差异程度
        diff_gray = np.mean(diff_map, axis=-1)

        # 6.3 应用高斯模糊，将“逐像素噪点差异”转化为“区域连片差异”
        # 核大小必须为奇数，默认 31 足够覆盖一定区域
        kernel_size = (args.blur_size, args.blur_size)
        diff_smoothed = cv2.GaussianBlur(diff_gray, kernel_size, 0)

        # 6.4 归一化差异，并计算残差权重映射图
        # 差异归一化到 0-1 之间，乘以敏感度因子
        diff_normalized = np.clip((diff_smoothed / 255.0) * args.diff_sensitivity, 0, 1)

        # 权重映射：差异越大，该值越接近 0；差异越小，该值越接近 1
        dynamic_weight_map = 1.0 - diff_normalized

        # 扩展维度以匹配图像通道数 (512, 512) -> (512, 512, 1)
        dynamic_weight_map = np.expand_dims(dynamic_weight_map, axis=-1)

        # 保存权重热力图以供调试 (越亮代表残差保留越多，越暗代表越少)
        weight_visual = (dynamic_weight_map[:, :, 0] * 255).astype(np.uint8)
        weight_save_path = os.path.join(args.save_path, "04_dynamic_weight_map.png")
        cv2.imwrite(weight_save_path, cv2.applyColorMap(weight_visual, cv2.COLORMAP_JET))
        print(f"--> 已保存残差动态权重热力图至: {weight_save_path}")

        # 6.5 应用动态权重进行相减
        # Test 的基础权重不变，残差的权重 = 基础权重 * 动态映射图
        final_residual_weight = args.weight2 * dynamic_weight_map
        test_subtracted_np = (args.weight1 * test_float) - (final_residual_weight * residual_float)

        # 6.6 降低整体亮度控制
        if args.brightness_factor != 1.0:
            test_subtracted_np *= args.brightness_factor
            print(f"--> 已应用亮度调整因子: {args.brightness_factor}")

        test_subtracted_clipped_np = np.clip(test_subtracted_np, 0, 255).astype(np.uint8)

        # 保存生成的相减图片
        test_subtracted_save_path = os.path.join(args.save_path, "04_test_minus_residual_dynamic.png")
        cv2.imwrite(test_subtracted_save_path, cv2.cvtColor(test_subtracted_clipped_np, cv2.COLOR_RGB2BGR))
        print(f"--> 已将 test_path 动态相减后的图片保存至: {test_subtracted_save_path}")

        # 测试准确率
        print("\n正在对 test_path 动态相减后的图片进行水印准确率检测...")
        try:
            test_subtracted_acc = evaluate_watermark_accuracy(test_subtracted_save_path)
            print("=" * 70)
            print(f"   [Test - 动态残差] 图片的水印提取准确率: {test_subtracted_acc:.4f}")
            print("=" * 70)
        except Exception as e:
            print(f"   [检测失败] 无法在 test_path 相减图片上提取水印，报错信息: {e}")

    print("流程结束。")