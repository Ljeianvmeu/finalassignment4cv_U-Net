# kmeans.py

import numpy as np
from sklearn.cluster import KMeans

def segment_image_kmeans(image_array, n_clusters=6):
    """
    对图像进行K-means聚类分割。

    参数：
    image_array (numpy.ndarray): 输入的图像数组，形状为 (H, W) 或 (H, W, C)。
    n_clusters (int): 聚类的类别数。

    返回：
    numpy.ndarray: 分割后的标签图像，形状为 (H, W)。
    """
    # 检查图像维度，如果是灰度图则扩展维度
    if image_array.ndim == 2:
        image_array = image_array[:, :, np.newaxis]

    h, w, c = image_array.shape
    # 将图像数据重塑为二维数组，每行代表一个像素的特征
    pixels = image_array.reshape(-1, c)

    # 进行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    # 将标签重塑回图像形状
    segmented_image = labels.reshape(h, w)
    return segmented_image
