import os
import sys
import numpy as np
from PIL import Image
import scipy.io
from matplotlib import cm
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

def pad_image(image, target_size=(256, 256), background_value=0):
    """
    将图像通过填充背景的方式扩充到指定大小。

    参数：
    image (numpy.ndarray): 输入的图像数组。
    target_size (tuple): 目标尺寸，默认 (256, 256)。
    background_value (int/float): 填充背景的像素值。

    返回：
    numpy.ndarray: 填充后的图像。
    """
    height, width = image.shape[:2]
    target_height, target_width = target_size

    pad_height = max(target_height - height, 0)
    pad_width = max(target_width - width, 0)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    if image.ndim == 2:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                              mode='constant', constant_values=background_value)
    else:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                              mode='constant', constant_values=background_value)
    return padded_image

def unpad_image(image, original_size):
    """
    将填充的图像还原到原始大小。

    参数：
    image (numpy.ndarray): 填充后的图像数组。
    original_size (tuple): 原始尺寸 (height, width)。

    返回：
    numpy.ndarray: 还原到原始大小的图像。
    """
    height, width = original_size
    current_height, current_width = image.shape[:2]

    pad_top = (current_height - height) // 2
    pad_left = (current_width - width) // 2

    return image[pad_top:pad_top+height, pad_left:pad_left+width]

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

    # 创建输出文件夹
    output_folder = os.path.splitext(mat_filename)[0] + '_output'
    os.makedirs(output_folder, exist_ok=True)
    # 创建子文件夹
    padded_images_folder = os.path.join(output_folder, 'padded_images')
    padded_labels_folder = os.path.join(output_folder, 'padded_labels')
    predicted_labels_folder = os.path.join(output_folder, 'predicted_labels')
    pseudocolor_folder = os.path.join(output_folder, 'pseudocolor_images')
    overlay_folder = os.path.join(output_folder, 'overlay_images')
    os.makedirs(padded_images_folder, exist_ok=True)
    os.makedirs(padded_labels_folder, exist_ok=True)
    os.makedirs(predicted_labels_folder, exist_ok=True)
    os.makedirs(pseudocolor_folder, exist_ok=True)
    os.makedirs(overlay_folder, exist_ok=True)

    # 定义模型
    unet = Unet()

    # 用于计算 mIoU
    miou_out_path = output_folder
    name_classes = ['background', 'class1', 'class2', 'class3', 'class4', 'class5']  # 根据实际类别修改
    num_classes = len(name_classes)

    # 保存真实标签和预测标签的路径，用于计算 mIoU
    gt_dir = os.path.join(output_folder, 'gt_labels')
    pred_dir = os.path.join(output_folder, 'predicted_labels')
    os.makedirs(gt_dir, exist_ok=True)

    # 用于保存每个切片的 mIoU
    slice_mious = []

    print("开始处理并预测切片...")
    for i in tqdm(range(num_slices)):
        # 提取第 i 张切片的图像和标签
        image_slice = images[:, :, i]
        label_slice = labels[:, :, i]

        # 保存原始尺寸
        slice_size = image_slice.shape

        # 获取背景值（例如，使用图像的最边缘像素值）
        background_value = image_slice[0, 0]

        # 对图像进行归一化，转换为 0-255 的 uint8 类型
        image_slice_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
        image_slice_normalized = image_slice_normalized.astype(np.uint8)

        # 对图像和标签进行填充到 256×256
        padded_image = pad_image(image_slice_normalized, target_size=(256, 256), background_value=background_value)
        padded_label = pad_image(label_slice, target_size=(256, 256), background_value=0)

        # 保存填充后的图像和标签，便于检查
        Image.fromarray(padded_image).save(os.path.join(padded_images_folder, f"slice_{i:03d}.png"))
        Image.fromarray(padded_label.astype(np.uint8)).save(os.path.join(padded_labels_folder, f"label_{i:03d}.png"))

        # 将填充后的图像转换为 PIL Image
        padded_image_pil = Image.fromarray(padded_image)

        # 进行模型预测
        predicted = unet.get_miou_png(padded_image_pil)

        # 将预测结果转换为 numpy 数组
        predicted_array = np.array(predicted)

        # 将预测结果还原到原始尺寸
        predicted_unpadded = unpad_image(predicted_array, original_size=slice_size)

        # 保存预测标签
        Image.fromarray(predicted_unpadded.astype(np.uint8)).save(os.path.join(predicted_labels_folder, f"label_{i:03d}.png"))

        # 保存真实标签（也截取为原始尺寸，以防万一）
        label_unpadded = unpad_image(padded_label, original_size=slice_size)
        Image.fromarray(label_unpadded.astype(np.uint8)).save(os.path.join(gt_dir, f"label_{i:03d}.png"))

        # 生成伪彩色图像
        colormap = cm.get_cmap('jet', num_classes)
        pseudocolor = colormap(predicted_unpadded / (num_classes - 1))  # 归一化
        pseudocolor = (pseudocolor[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(pseudocolor).save(os.path.join(pseudocolor_folder, f"pseudocolor_{i:03d}.png"))

        # 生成叠加图
        original_image = Image.fromarray(image_slice_normalized)
        original_image = original_image.convert('RGBA')
        pseudocolor_image = Image.fromarray(pseudocolor)
        pseudocolor_image = pseudocolor_image.convert('RGBA')
        overlay_image = Image.blend(original_image, pseudocolor_image, alpha=0.5)
        overlay_image.save(os.path.join(overlay_folder, f"overlay_{i:03d}.png"))

        # 计算当前切片的 mIoU
        # 由于切片尺寸较小，可以直接计算
        intersection = np.logical_and(label_unpadded == predicted_unpadded, label_unpadded > 0)
        union = np.logical_or(label_unpadded > 0, predicted_unpadded > 0)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 1.0
        slice_mious.append(iou)

    print("切片预测完成，开始计算 mIoU...")

    # 计算整体的 mIoU
    image_ids = [f"label_{i:03d}" for i in range(num_slices)]
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)

    # 保存 mIoU 结果
    results_file = os.path.join(output_folder, 'miou_results.txt')
    with open(results_file, 'w') as f:
        f.write("Per-slice mIoUs:\n")
        for i, iou in enumerate(slice_mious):
            f.write(f"Slice {i:03d}: mIoU = {iou:.4f}\n")
        f.write("\nOverall mIoU:\n")
        for i in range(num_classes):
            f.write(f"Class {name_classes[i]}: IoU = {IoUs[i]:.4f}\n")
        f.write(f"\nMean mIoU: {np.nanmean(IoUs):.4f}\n")

    print("mIoU 计算完成，结果已保存。")

if __name__ == "__main__":
    main()
