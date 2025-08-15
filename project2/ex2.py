import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import sys

try:
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    # 如果在某些环境中无法获取路径，使用当前工作目录
    current_dir = os.getcwd()

# 定义文件路径
host_path = os.path.join(current_dir, 'host.jpg')
watermark_path = os.path.join(current_dir, 'watermark.png')
watermarked_path = os.path.join(current_dir, 'watermarked.jpg')
extracted_path = os.path.join(current_dir, 'extracted.png')
results_dir = os.path.join(current_dir, 'results')


def generate_sample_images():
    """生成示例图像（如果用户没有提供）"""
    print("正在生成示例图像...")

    # 创建宿主图像
    host = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(host, 'Sample Host Image', (50, 256), cv2.FONT_HERSHEY_SIMPLEX,
                1, 255, 2, cv2.LINE_AA)
    cv2.imwrite(host_path, host)

    # 创建水印图像
    watermark = np.zeros((128, 128), dtype=np.uint8)
    cv2.putText(watermark, 'WATERMARK', (10, 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 255, 1, cv2.LINE_AA)
    cv2.imwrite(watermark_path, watermark)

    print(f"已生成示例图像: {host_path} 和 {watermark_path}")


def embed_watermark(host_img, watermark, alpha=0.05):
    """嵌入水印到宿主图像中"""
    # 小波变换
    coeffs = pywt.dwt2(host_img, 'haar')
    LL, (LH, HL, HH) = coeffs

    # 调整水印尺寸
    watermark_resized = cv2.resize(watermark, (LL.shape[1], LL.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    watermark_binary = (watermark_resized > 128).astype(np.float32)

    # 嵌入水印到低频分量
    LL_watermarked = LL + alpha * watermark_binary * np.max(LL)

    # 逆小波变换
    coeffs_watermarked = (LL_watermarked, (LH, HL, HH))
    img_watermarked = pywt.idwt2(coeffs_watermarked, 'haar')

    # 处理边界并转换为uint8
    img_watermarked = np.clip(img_watermarked, 0, 255)
    return img_watermarked.astype(np.uint8)


def extract_watermark(watermarked_img, original_img, alpha=0.05):
    """从含水印图像中提取水印"""
    # 原始图像小波变换
    coeffs_orig = pywt.dwt2(original_img, 'haar')
    LL_orig, _ = coeffs_orig

    # 含水印图像小波变换
    coeffs_wm = pywt.dwt2(watermarked_img, 'haar')
    LL_wm, _ = coeffs_wm

    # 提取水印
    watermark_extracted = (LL_wm - LL_orig) / (alpha * np.max(LL_orig))

    # 二值化处理
    watermark_extracted = np.clip(watermark_extracted, 0, 1)
    watermark_extracted = (watermark_extracted * 255).astype(np.uint8)

    return watermark_extracted


def calculate_nc(orig_wm, extracted_wm):
    """计算归一化相关系数(NC)"""
    # 调整到相同尺寸
    h, w = orig_wm.shape
    extracted_wm = cv2.resize(extracted_wm, (w, h))

    # 归一化处理
    orig_norm = orig_wm.astype(np.float32) / 255.0
    extr_norm = extracted_wm.astype(np.float32) / 255.0

    # 计算NC
    numerator = np.sum(orig_norm * extr_norm)
    denominator = np.sqrt(np.sum(orig_norm ** 2)) * np.sqrt(np.sum(extr_norm ** 2))

    if denominator == 0:
        return 0
    return numerator / denominator


def add_salt_pepper_noise(image, prob=0.05):
    """添加椒盐噪声"""
    noisy = np.copy(image)
    # 添加盐噪声 (白点)
    salt_mask = np.random.random(image.shape[:2]) < prob / 2
    noisy[salt_mask] = 255
    # 添加椒噪声 (黑点)
    pepper_mask = np.random.random(image.shape[:2]) < prob / 2
    noisy[pepper_mask] = 0
    return noisy


def apply_attacks(image):
    """应用各种攻击并返回结果字典"""
    attacked_images = {}

    # 旋转攻击 (15度)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 15, 1)
    attacked_images['旋转15度'] = cv2.warpAffine(image, M, image.shape[::-1])

    # 平移攻击 (30像素)
    M = np.float32([[1, 0, 30], [0, 1, 30]])
    attacked_images['平移(30,30)'] = cv2.warpAffine(image, M, image.shape[::-1])

    # 裁剪攻击 (10%)
    h, w = image.shape
    cropped = image[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    attacked_images['裁剪10%'] = cv2.resize(cropped, (w, h))

    # 对比度增强
    attacked_images['对比度增强'] = np.clip(image.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

    # 椒盐噪声
    attacked_images['椒盐噪声'] = add_salt_pepper_noise(image, 0.05)

    # 高斯模糊
    attacked_images['高斯模糊'] = cv2.GaussianBlur(image, (5, 5), 0)

    # JPEG压缩 (模拟)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    attacked_images['JPEG压缩'] = cv2.imdecode(encimg, 0)

    # 缩放攻击
    scaled = cv2.resize(image, (int(w * 0.8), int(h * 0.8)))
    attacked_images['缩放80%'] = cv2.resize(scaled, (w, h))

    # 亮度调整
    attacked_images['亮度增加'] = np.clip(image.astype(np.float32) + 50, 0, 255).astype(np.uint8)

    return attacked_images


def robustness_test(watermarked_img, host_img, watermark_img, alpha=0.05):
    """测试水印对各种攻击的鲁棒性"""
    # 应用攻击
    attacked_images = apply_attacks(watermarked_img)

    results = {}

    # 创建输出目录
    os.makedirs(results_dir, exist_ok=True)

    # 保存原始水印
    cv2.imwrite(os.path.join(results_dir, 'original_watermark.png'), watermark_img)

    # 测试每种攻击
    for attack_name, attacked_img in attacked_images.items():
        try:
            # 保存攻击后的图像
            attack_path = os.path.join(results_dir, f'{attack_name}.png')
            cv2.imwrite(attack_path, attacked_img)

            # 提取水印
            extracted_wm = extract_watermark(attacked_img, host_img, alpha)
            extracted_path = os.path.join(results_dir, f'{attack_name}_extracted.png')
            cv2.imwrite(extracted_path, extracted_wm)

            # 计算NC值
            nc = calculate_nc(watermark_img, extracted_wm)
            results[attack_name] = nc

        except Exception as e:
            print(f"处理攻击 '{attack_name}' 时出错: {e}")
            results[attack_name] = 0

    # 可视化结果
    try:
        plt.figure(figsize=(15, 12))
        for i, (attack_name, _) in enumerate(attacked_images.items()):
            extracted_path = os.path.join(results_dir, f'{attack_name}_extracted.png')
            if os.path.exists(extracted_path):
                extracted_wm = cv2.imread(extracted_path, 0)
                plt.subplot(4, 3, i + 1)
                plt.imshow(extracted_wm, cmap='gray')
                plt.title(f'{attack_name}\nNC={results[attack_name]:.4f}')
                plt.axis('off')

        # 添加原始水印对比
        plt.subplot(4, 3, 12)
        plt.imshow(watermark_img, cmap='gray')
        plt.title('原始水印')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'robustness_test_results.png'))
        plt.show()
    except Exception as e:
        print(f"可视化时出错: {e}")

    return results


def plot_watermark_comparison(host, watermarked, watermark, extracted):
    """可视化水印嵌入前后对比"""
    try:
        plt.figure(figsize=(15, 10))

        # 原始宿主图像
        plt.subplot(2, 3, 1)
        plt.imshow(host, cmap='gray')
        plt.title('原始宿主图像')
        plt.axis('off')

        # 含水印图像
        plt.subplot(2, 3, 2)
        plt.imshow(watermarked, cmap='gray')
        plt.title('含水印图像')
        plt.axis('off')

        # 差异图 (放大10倍便于观察)
        diff = np.abs(host.astype(np.float32) - watermarked.astype(np.float32))
        plt.subplot(2, 3, 3)
        plt.imshow(diff * 10, cmap='hot')
        plt.title('差异图 (10x放大)')
        plt.axis('off')
        plt.colorbar()

        # 原始水印
        plt.subplot(2, 3, 4)
        plt.imshow(watermark, cmap='gray')
        plt.title('原始水印')
        plt.axis('off')

        # 提取的水印
        plt.subplot(2, 3, 5)
        plt.imshow(extracted, cmap='gray')
        plt.title('提取的水印')
        plt.axis('off')

        # 水印差异
        wm_diff = np.abs(watermark.astype(np.float32) - cv2.resize(extracted, watermark.shape[::-1]).astype(np.float32))
        plt.subplot(2, 3, 6)
        plt.imshow(wm_diff, cmap='hot')
        plt.title('水印差异')
        plt.axis('off')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'watermark_comparison.png'))
        plt.show()
    except Exception as e:
        print(f"可视化比较时出错: {e}")


def main():
    print(f"当前工作目录: {current_dir}")

    # 检查并生成示例图像（如果不存在）
    if not os.path.exists(host_path) or not os.path.exists(watermark_path):
        print("未找到宿主图像或水印图像，正在生成示例图像...")
        generate_sample_images()

    # 读取图像
    try:
        print(f"尝试从 {host_path} 加载宿主图像")
        host = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)

        print(f"尝试从 {watermark_path} 加载水印图像")
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"读取图像时出错: {e}")
        return

    if host is None:
        print(f"错误: 无法加载宿主图像 '{host_path}'")
        print("可能原因: 文件不存在、文件损坏或权限问题")
        return

    if watermark is None:
        print(f"错误: 无法加载水印图像 '{watermark_path}'")
        print("可能原因: 文件不存在、文件损坏或权限问题")
        return

    # 检查图像尺寸
    print(f"宿主图像尺寸: {host.shape}, 水印图像尺寸: {watermark.shape}")

    # 嵌入水印
    try:
        watermarked = embed_watermark(host, watermark, alpha=0.1)
        cv2.imwrite(watermarked_path, watermarked)
        print(f"已生成含水印图像: {watermarked_path}")
    except Exception as e:
        print(f"嵌入水印时出错: {e}")
        return

    # 计算PSNR
    try:
        mse = np.mean((host - watermarked) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
        print(f"PSNR (嵌入质量): {psnr:.2f} dB")
    except:
        print("无法计算PSNR")

    # 提取水印 (未攻击)
    try:
        extracted = extract_watermark(watermarked, host)
        cv2.imwrite(extracted_path, extracted)
        print(f"已提取水印: {extracted_path}")
    except Exception as e:
        print(f"提取水印时出错: {e}")
        return

    # 计算原始NC值
    try:
        orig_nc = calculate_nc(watermark, extracted)
        print(f"原始NC值: {orig_nc:.4f}")
    except:
        print("无法计算NC值")

    # 可视化比较
    plot_watermark_comparison(host, watermarked, watermark, extracted)

    # 鲁棒性测试
    print("\n进行鲁棒性测试...")
    try:
        test_results = robustness_test(watermarked, host, watermark)

        # 打印测试结果
        print("\n鲁棒性测试结果:")
        print("-" * 40)
        print(f"{'攻击类型':<20} | {'NC值':<10}")
        print("-" * 40)
        for attack, nc in test_results.items():
            print(f"{attack:<20} | {nc:.4f}")
        print("-" * 40)
    except Exception as e:
        print(f"鲁棒性测试失败: {e}")


if __name__ == "__main__":
    # 检查并安装必要的库
    try:
        import cv2
        import numpy as np
        import pywt
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("正在尝试安装所需库...")
        try:
            import pip

            pip.main(['install', 'opencv-python', 'numpy', 'pywavelets', 'matplotlib', 'scikit-image'])
            print("库安装成功，请重新运行程序。")
        except:
            print("自动安装失败，请手动运行以下命令:")
            print("pip install opencv-python numpy pywavelets matplotlib scikit-image")
        sys.exit(1)

    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)

    main()
    print(f"程序执行完成。结果保存在 '{results_dir}' 目录中。")
    print(f"生成的图像文件: ")
    print(f" - 宿主图像: {host_path}")
    print(f" - 水印图像: {watermark_path}")
    print(f" - 含水印图像: {watermarked_path}")
    print(f" - 提取的水印: {extracted_path}")