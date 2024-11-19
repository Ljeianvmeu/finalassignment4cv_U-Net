# improved_kmeans.py

import os
import sys
import numpy as np
from PIL import Image
import scipy.io
from matplotlib import cm
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import cv2  # OpenCV 库用于图像处理

from utils.utils_metrics import compute_mIoU, show_results

def remap_labels(predicted_labels, true_labels, num_classes):
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
    name_classes = ['background', 'skin/scalp', 'skull', 'CSF', 'Gray Matter', 'White Matter']  # 根据实际类别修改
    num_classes = len(name_classes)

    # 创建输出根目录
    output_root = 'output'
    os.makedirs(output_root, exist_ok=True)

    # 创建输出文件夹
    kmeans_output_folder = os.path.join(output_root, 'improved_kmeans_' + os.path.splitext(mat_filename)[0] + '_output')
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

    print("开始使用改进的 K-means 进行预测...")

    for i in tqdm(range(num_slices)):
        # 提取第 i 张切片的图像和标签
        image_slice = images[:, :, i]
        label_slice = labels[:, :, i]

        # 对图像进行归一化，转换为 0-255 的 uint8 类型
        image_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
        image_normalized = image_normalized.astype(np.uint8)

        # 使用高斯滤波进行平滑
        image_smoothed = cv2.GaussianBlur(image_normalized, (5, 5), 0)

        # 使用 Canny 边缘检测
        edges = cv2.Canny(image_smoothed, threshold1=50, threshold2=150)

        # 构建特征向量，包括像素值和边缘信息
        h, w = image_normalized.shape
        pixels = image_normalized.flatten()
        edges_flat = edges.flatten()
        features = np.stack((pixels, edges_flat), axis=1)

        # 进行 K-means 分割
        kmeans = KMeans(n_clusters=num_classes, random_state=0)
        kmeans.fit(features)
        predicted_labels = kmeans.labels_.reshape(h, w)

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
        original_image = Image.fromarray(image_normalized).convert('RGBA')
        pseudocolor_image = Image.fromarray(pseudocolor).convert('RGBA')
        overlay_image = Image.blend(original_image, pseudocolor_image, alpha=0.5)
        overlay_image.save(os.path.join(kmeans_overlay_folder, f"overlay_{i:03d}.png"))

    print("改进的 K-means 预测完成，开始计算 mIoU...")

    # 计算整体的 mIoU
    image_ids = [f"label_{i:03d}" for i in range(num_slices)]
    hist, IoUs, PA_Recall, Precision = compute_mIoU(kmeans_gt_dir, kmeans_predicted_labels_folder, image_ids, num_classes, name_classes)

    # 保存 mIoU 结果
    results_file = os.path.join(kmeans_output_folder, 'miou_results.txt')
    with open(results_file, 'w') as f:
        f.write("\nImproved K-means Prediction Overall mIoU:\n")
        for idx in range(num_classes):
            f.write(f"Class {name_classes[idx]}: IoU = {IoUs[idx]:.4f}\n")
        f.write(f"\nMean mIoU: {np.nanmean(IoUs):.4f}\n")

    print("改进的 K-means mIoU 计算完成，结果已保存。")

if __name__ == "__main__":
    main()
