
import os

# 导入检测水印的水印核心函数
try:
    from decode_image import evaluate_watermark_accuracy
except ImportError:
    print("错误: 未找到 decode_image 模块或 evaluate_watermark_accuracy 函数，请确保文件在当前路径或 PYTHONPATH 中。")
    exit(1)

if __name__ == "__main__":
    # 在列表中填入所有需要测试的图片路径
    image_paths = [
        "/opt/data/jiang/cross/CRoSS/image_jiang/gauss/13/02_extra_image_B.png",
        "/opt/data/jiang/cross/CRoSS/image_jiang/gauss/13/02_hidden_image_A_residual.png",
        "/opt/data/jiang/cross/CRoSS/image_jiang/gauss/13/02_extra_D1_minus_D2.png",
    ]

    print("=========================================")
    print(f"开始批量检测，共 {len(image_paths)} 张图片")
    print("=========================================\n")

    for path in image_paths:
        print(f"正在检测: {path}")

        # 检查图片是否存在
        if not os.path.exists(path):
            print(f"-> [错误] 找不到文件，已跳过该图片。\n")
            continue

        # 执行水印准确率评估
        try:
            acc = evaluate_watermark_accuracy(path)
            print(f"-> [成功] 提取准确率: {acc:.4f} ({acc * 100:.2f}%)\n")
        except Exception as e:
            print(f"-> [报错] 检测过程中发生错误: {e}\n")