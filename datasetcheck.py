# check_data_overlap.py

import os
import sys
import numpy as np
from PIL import Image
import scipy.io
import imagehash
from tqdm import tqdm

def load_training_images(training_images_dir):
    """
    加载训练数据集中的图像，并计算其感知哈希值。

    参数：
    training_images_dir (str): 训练图像的目录路径。

    返回：
    dict: 训练图像的哈希值字典，键为哈希值，值为图像文件路径。
    """
    training_hashes = {}
    i = 0
    print("正在加载训练图像并计算哈希值...")
    for root, dirs, files in os.walk(training_images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                try:
                    # 加载图像并转换为灰度图像
                    image = Image.open(image_path).convert('L')
                    # 计算图像的感知哈希值
                    hash_value = imagehash.phash(image)
                    # 存储哈希值和对应的图像路径
                    training_hashes[str(hash_value)] = image_path
                    i += 1
                except Exception as e:
                    print(f"加载图像 {image_path} 时出错：{e}")
    print(f"共加载了 {i} 张训练图像。" + f"其中具有哈希值的图像有{len(training_hashes)}张。")
    return training_hashes

def main():
    # 输入测试数据的 .mat 文件名称
    mat_filename = input("请输入测试数据的 .mat 文件名称（包含扩展名，例如 'Brain.mat'）：").strip()

    # 检查文件是否存在
    if not os.path.exists(mat_filename):
        print(f"文件 {mat_filename} 不存在，请检查文件路径。")
        sys.exit(1)

    # 加载 .mat 文件
    data = scipy.io.loadmat(mat_filename)

    # 提取图像数据
    if 'T1' not in data:
        print("无法在 .mat 文件中找到键 'T1'。")
        sys.exit(1)
    images = data['T1']  # 假设图像数据存储在键 'T1' 中
    num_slices = images.shape[2]

    # 输入训练图像的目录路径
    training_images_dir = input("请输入训练图像的目录路径：").strip()
    if not os.path.isdir(training_images_dir):
        print(f"目录 {training_images_dir} 不存在，请检查路径。")
        sys.exit(1)

    # 加载训练图像并计算哈希值
    training_hashes = load_training_images(training_images_dir)

    # 设置哈希差异阈值
    hash_threshold = 5  # 可根据需要调整

    # 比较测试图像与训练图像
    print("开始比较测试图像与训练图像...")
    duplicates = []
    for i in tqdm(range(num_slices)):
        image_slice = images[:, :, i]
        # 归一化并转换为 0-255 的 uint8 类型
        image_slice_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
        image_slice_normalized = image_slice_normalized.astype(np.uint8)
        # 将 numpy 数组转换为 PIL 图像
        image_pil = Image.fromarray(image_slice_normalized).convert('L')
        # 计算测试图像的感知哈希值
        test_hash = imagehash.phash(image_pil)
        test_hash_str = str(test_hash)

        # 初始化匹配标志
        match_found = False

        # 精确匹配检查
        if test_hash_str in training_hashes:
            match_found = True
            duplicates.append((i, training_hashes[test_hash_str], 0))
            print(f"测试图像切片 {i} 与训练图像精确匹配：{training_hashes[test_hash_str]}")
        else:
            # 遍历训练图像哈希，计算哈希差异
            for train_hash_str, train_image_path in training_hashes.items():
                hash_diff = test_hash - imagehash.hex_to_hash(train_hash_str)
                if hash_diff < hash_threshold:
                    match_found = True
                    duplicates.append((i, train_image_path, hash_diff))
                    print(f"测试图像切片 {i} 与训练图像 {train_image_path} 相似，哈希差异为 {hash_diff}")
                    break  # 如果找到相似的，跳出循环
            if not match_found:
                print(f"测试图像切片 {i} 未在训练集中找到匹配。")

    # 输出匹配结果
    if duplicates:
        print("\n检测到测试数据与训练数据集中存在以下重复或相似的图像：")
        for slice_idx, train_image_path, hash_diff in duplicates:
            print(f"测试切片 {slice_idx} - 训练图像：{train_image_path} - 哈希差异：{hash_diff}")
    else:
        print("\n未发现测试数据与训练数据集中有重复的图像。")

if __name__ == "__main__":
    main()
