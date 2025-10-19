import os
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error as mse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 初始化 CLIP 模型和处理器
clip_model = CLIPModel.from_pretrained('')
clip_processor = CLIPProcessor.from_pretrained('')

def load_png_image(png_file):
    """从 PNG 文件加载图像"""
    image = Image.open(png_file).convert('RGB')  # 转换为 RGB 图像
    return np.array(image) / 255.0  # 归一化到 [0, 1]

def calculate_clip_similarity(image1_path, image2_path):
    """计算两张图片之间的CLIP相似性"""
    image1 = Image.open(image1_path).convert('RGB')
    inputs1 = clip_processor(images=image1, return_tensors='pt')

    image2 = Image.open(image2_path).convert('RGB')
    inputs2 = clip_processor(images=image2, return_tensors='pt')

    with torch.no_grad():
        image1_features = clip_model.get_image_features(inputs1.pixel_values)
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features = clip_model.get_image_features(inputs2.pixel_values)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        # 计算相似度（余弦相似度）
        similarity = torch.matmul(image1_features, image2_features.T).item()

    return similarity

def calculate_metrics(image1, image2):
    # 将图像转换为浮点型
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    if image1.ndim == 2:  # 灰度图像
        image1 = np.stack([image1] * 3, axis=-1)  # 重复三个通道
    if image2.ndim == 2:  # 灰度图像
        image2 = np.stack([image2] * 3, axis=-1)  # 重复三个通道

    # 计算 PSNR
    psnr_value = psnr(image1, image2, data_range=1.0)

    # 计算 NCC
    image1_gray = rgb2gray(image1)
    image2_gray = rgb2gray(image2)
    ncc_value = np.corrcoef(image1_gray.flatten(), image2_gray.flatten())[0, 1]

    # 计算 SSIM
    ssim_value = ssim(image1, image2, multichannel=True, data_range=1.0, win_size=3)

    # 计算 LPIPS
    loss_fn = lpips.LPIPS(net='alex')  # 选择网络，可以选择 'alex', 'vgg', 'squeeze'
    image1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    image2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    lpips_value = loss_fn(image1_tensor, image2_tensor)

    # 计算 MSE
    mse_value = mse(image1, image2)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'NCC': ncc_value,
        'LPIPS': lpips_value.item(),
        'MSE': mse_value
    }

def process_image_pairs(gt_folder, sim_folder, output_file):
    """计算两个文件夹中成对图片的指标，并将结果写入文件"""
    gt_images = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])
    sim_images = sorted([f for f in os.listdir(sim_folder) if f.endswith('.png')])

    if len(gt_images) != len(sim_images):
        raise ValueError("两个文件夹中的图片数量不一致！")

    # 设置固定列宽
    col_widths = [20, 10, 10, 10, 10, 10, 10]  # 对应列的宽度
    header = ["Image_Name", "PSNR", "SSIM", "NCC", "LPIPS", "MSE", "CLIP"]

    metrics_sum = {key: 0.0 for key in ['PSNR', 'SSIM', 'NCC', 'LPIPS', 'MSE', 'CLIP']}

    with open(output_file, 'w') as file:
        # 写入标题行
        file.write("".join(f"{h:<{w}}" for h, w in zip(header, col_widths)) + "\n")

        for gt_image, sim_image in zip(gt_images, sim_images):
            gt_path = os.path.join(gt_folder, gt_image)
            sim_path = os.path.join(sim_folder, sim_image)

            gt = load_png_image(gt_path)
            sim = load_png_image(sim_path)
            metrics = calculate_metrics(gt, sim)

            # 计算 CLIP 相似性
            clip_similarity = calculate_clip_similarity(gt_path, sim_path)
            metrics['CLIP'] = clip_similarity

            # 累加指标值
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]

            # 写入结果到文件
            file.write(f"{sim_image:<20}\t{metrics['PSNR']:.4f}\t{metrics['SSIM']:.4f}\t{metrics['NCC']:.4f}\t{metrics['LPIPS']:.4f}\t{metrics['MSE']:.4f}\t{clip_similarity:.4f}\n")

        # 计算均值
        num_images = len(gt_images)
        metrics_avg = {key: metrics_sum[key] / num_images for key in metrics_sum}

        # 写入均值到文件
        file.write("\n" + "Average:".ljust(20) + "\t".join(f"{metrics_avg[key]:.4f}" for key in metrics_avg) + "\n")

# 示例用法
gt_folder = ''
sim_folder = ''
output_file = '\\metrics.txt'

process_image_pairs(gt_folder, sim_folder, output_file)
print(f"指标已保存至 {output_file}")

