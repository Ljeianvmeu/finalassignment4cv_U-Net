# predict.py

import os
import sys
import numpy as np
from PIL import Image
import scipy.io
from matplotlib import cm
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

# 导入 kmeans.py 中的函数
from kmeans import segment_image_kmeans

# 添加 remap_labels 函数
from scipy.optimize import linear_sum_assignment

def remap_labels(predicted_labels, true_labels, num_classes):
    from sklearn.metrics import confusion_matrix

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels.flatten(), predicted_labels.flatten(), labels=range(num_classes))

    # 匈牙利算法求最佳匹配
    row_ind, col_ind = linear_sum_assignment(-cm)  # 负号是因为该函数求最小化

    # 创建标签映射表
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

    # 进行标签映射
    remapped_labels = np.copy(predicted_labels)
    for old_label, new_label in mapping.items():
        remapped_labels[predicted_labels == old_label] = new_label

    return remapped_labels

def main():
    # 从控制台输入 .mat 文件名称
    mat_filename = input("请输入测试数据的 .mat 文件名称（包含扩展名，例如 'Brain.mat'）：")

    # 检查文件是否存在
    if not os.path.exists(mat_filename):
        print(f"文件 {mat_filename} 不存在，请检查文件路径。")
        sys.exit(1)

    # 加载 .mat 文件
    data = scipy.io.loadmat(mat_filename)

    # 提取图像和标签数据
    # 假设图像数据存储在键 'T1'，标签数据存储在键 'label' 中
    if 'T1' not in data or 'label' not in data:
        print("无法在 .mat 文件中找到 'T1' 或 'label' 数据。")
        sys.exit(1)

    images = data['T1']       # 图像数据，形状为 (height, width, num_slices)
    labels = data['label']    # 标签数据，形状为 (height, width, num_slices)

    # 检查图像和标签的尺寸是否匹配
    if images.shape != labels.shape:
        print("图像和标签的尺寸不匹配。")
        sys.exit(1)

    num_slices = images.shape[2]
    original_size = images.shape[:2]  # (height, width)

    # 定义类别信息
    name_classes = ['background', 'class1', 'class2', 'class3', 'class4', 'class5']  # 根据实际类别修改
    num_classes = len(name_classes)

    # 创建输出根目录
    output_root = 'output'
    os.makedirs(output_root, exist_ok=True)

    # -------------------- 使用 Unet 进行预测 --------------------
    # 创建输出文件夹
    unet_output_folder = os.path.join(output_root, 'unet_' + os.path.splitext(mat_filename)[0] + '_output')
    os.makedirs(unet_output_folder, exist_ok=True)
    # 创建子文件夹
    unet_predicted_labels_folder = os.path.join(unet_output_folder, 'predicted_labels')
    unet_pseudocolor_folder = os.path.join(unet_output_folder, 'pseudocolor_images')
    unet_overlay_folder = os.path.join(unet_output_folder, 'overlay_images')
    os.makedirs(unet_predicted_labels_folder, exist_ok=True)
    os.makedirs(unet_pseudocolor_folder, exist_ok=True)
    os.makedirs(unet_overlay_folder, exist_ok=True)

    # 定义模型
    unet = Unet()

    # 保存真实标签和预测标签的路径，用于计算 mIoU
    unet_gt_dir = os.path.join(unet_output_folder, 'gt_labels')
    os.makedirs(unet_gt_dir, exist_ok=True)

    print("开始使用 Unet 进行预测...")
    for i in tqdm(range(num_slices)):
        # 提取第 i 张切片的图像和标签
        image_slice = images[:, :, i]
        label_slice = labels[:, :, i]

        # 对图像进行归一化，转换为 0-255 的 uint8 类型
        image_slice_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
        image_slice_normalized = image_slice_normalized.astype(np.uint8)

        # 将图像转换为 PIL Image
        image_pil = Image.fromarray(image_slice_normalized)

        # 进行模型预测
        predicted = unet.get_miou_png(image_pil)

        # 将预测结果转换为 numpy 数组
        predicted_array = np.array(predicted)

        # 保存预测标签
        Image.fromarray(predicted_array.astype(np.uint8)).save(os.path.join(unet_predicted_labels_folder, f"label_{i:03d}.png"))

        # 保存真实标签
        Image.fromarray(label_slice.astype(np.uint8)).save(os.path.join(unet_gt_dir, f"label_{i:03d}.png"))

        # 生成伪彩色图像
        colormap = cm.get_cmap('jet', num_classes)
        pseudocolor = colormap(predicted_array / (num_classes - 1))  # 归一化
        pseudocolor = (pseudocolor[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(pseudocolor).save(os.path.join(unet_pseudocolor_folder, f"pseudocolor_{i:03d}.png"))

        # 生成叠加图
        original_image = image_pil.convert('RGBA')
        pseudocolor_image = Image.fromarray(pseudocolor).convert('RGBA')
        overlay_image = Image.blend(original_image, pseudocolor_image, alpha=0.5)
        overlay_image.save(os.path.join(unet_overlay_folder, f"overlay_{i:03d}.png"))

    print("Unet 预测完成，开始计算 mIoU...")

    # 计算整体的 mIoU
    image_ids = [f"label_{i:03d}" for i in range(num_slices)]
    hist, IoUs, PA_Recall, Precision = compute_mIoU(unet_gt_dir, unet_predicted_labels_folder, image_ids, num_classes, name_classes)

    # 保存 mIoU 结果
    results_file = os.path.join(unet_output_folder, 'miou_results.txt')
    with open(results_file, 'w') as f:
        f.write("\nUnet Prediction Overall mIoU:\n")
        for idx in range(num_classes):
            f.write(f"Class {name_classes[idx]}: IoU = {IoUs[idx]:.4f}\n")
        f.write(f"\nMean mIoU: {np.nanmean(IoUs):.4f}\n")

    print("Unet mIoU 计算完成，结果已保存。")

    # -------------------- 使用 K-means 进行预测 --------------------
    # 创建输出文件夹
    kmeans_output_folder = os.path.join(output_root, 'kmeans_' + os.path.splitext(mat_filename)[0] + '_output')
    os.makedirs(kmeans_output_folder, exist_ok=True)
    # 创建子文件夹
    kmeans_predicted_labels_folder = os.path.join(kmeans_output_folder, 'predicted_labels')
    kmeans_pseudocolor_folder = os.path.join(kmeans_output_folder, 'pseudocolor_images')
    kmeans_overlay_folder = os.path.join(kmeans_output_folder, 'overlay_images')
    os.makedirs(kmeans_predicted_labels_folder, exist_ok=True)
    os.makedirs(kmeans_pseudocolor_folder, exist_ok=True)
    os.makedirs(kmeans_overlay_folder, exist_ok=True)

    # 保存真实标签和预测标签的路径，用于计算 mIoU
    kmeans_gt_dir = os.path.join(kmeans_output_folder, 'gt_labels')
    os.makedirs(kmeans_gt_dir, exist_ok=True)

    print("开始使用 K-means 进行预测...")
    for i in tqdm(range(num_slices)):
        # 提取第 i 张切片的图像和标签
        image_slice = images[:, :, i]
        label_slice = labels[:, :, i]

        # 对图像进行归一化，转换为 0-255 的 uint8 类型
        image_slice_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
        image_slice_normalized = image_slice_normalized.astype(np.uint8)

        # 进行 K-means 分割，不添加空间信息
        predicted_labels = segment_image_kmeans(image_slice_normalized, n_clusters=num_classes)

        # 对预测结果进行标签映射
        remapped_labels = remap_labels(predicted_labels, label_slice, num_classes)

        # 保存预测标签
        Image.fromarray(remapped_labels.astype(np.uint8)).save(os.path.join(kmeans_predicted_labels_folder, f"label_{i:03d}.png"))

        # 保存真实标签
        Image.fromarray(label_slice.astype(np.uint8)).save(os.path.join(kmeans_gt_dir, f"label_{i:03d}.png"))

        # 生成伪彩色图像
        colormap = cm.get_cmap('jet', num_classes)
        pseudocolor = colormap(remapped_labels / (num_classes - 1))  # 归一化
        pseudocolor = (pseudocolor[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(pseudocolor).save(os.path.join(kmeans_pseudocolor_folder, f"pseudocolor_{i:03d}.png"))

        # 生成叠加图
        original_image = Image.fromarray(image_slice_normalized).convert('RGBA')
        pseudocolor_image = Image.fromarray(pseudocolor).convert('RGBA')
        overlay_image = Image.blend(original_image, pseudocolor_image, alpha=0.5)
        overlay_image.save(os.path.join(kmeans_overlay_folder, f"overlay_{i:03d}.png"))

    print("K-means 预测完成，开始计算 mIoU...")

    # 计算整体的 mIoU
    image_ids = [f"label_{i:03d}" for i in range(num_slices)]
    hist, IoUs, PA_Recall, Precision = compute_mIoU(kmeans_gt_dir, kmeans_predicted_labels_folder, image_ids, num_classes, name_classes)

    # 保存 mIoU 结果
    results_file = os.path.join(kmeans_output_folder, 'miou_results.txt')
    with open(results_file, 'w') as f:
        f.write("\nK-means Prediction Overall mIoU:\n")
        for idx in range(num_classes):
            f.write(f"Class {name_classes[idx]}: IoU = {IoUs[idx]:.4f}\n")
        f.write(f"\nMean mIoU: {np.nanmean(IoUs):.4f}\n")

    print("K-means mIoU 计算完成，结果已保存。")

if __name__ == "__main__":
    main()
