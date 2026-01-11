from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2  # OpenCV 用于边缘检测
import matplotlib

matplotlib.use('TkAgg')  # 使用 TkAgg 后端

def simulate_corrosion_from_edge(image_path, num_steps=50, max_corrosion_size=50, random_seed=42):
    # 打开图像并转换为 RGBA 以支持透明背景
    image = Image.open(image_path)
    image = image.convert("RGBA")

    # 将图像转为 numpy 数组
    img_array = np.array(image)

    # 获取图像的宽高
    height, width, _ = img_array.shape

    # 设置随机种子，确保结果可复现
    random.seed(random_seed)

    # 使用 OpenCV 进行边缘检测，找出图像的边缘
    alpha_channel = img_array[:, :, 3]  # 提取 alpha 通道（透明度）
    edges = cv2.Canny(alpha_channel, 100, 200)  # 通过 Canny 边缘检测算法提取边缘

    # 创建一个列表保存每个阶段的腐蚀效果图像
    corrosion_images = []

    # 多层腐蚀效果，模拟逐层扩展的腐蚀
    # 使用指数衰减函数来平滑腐蚀宽度
    corrosion_widths = np.logspace(0, np.log10(max_corrosion_size), num_steps)

    # 腐蚀过程
    for step in range(num_steps):
        # 创建一个副本来进行腐蚀操作
        corrupted_image = np.copy(img_array)

        # 将 numpy 数组转换回 PIL 图像
        pil_corrupted_image = Image.fromarray(corrupted_image)

        # 当前步骤的腐蚀宽度
        corrosion_width = int(corrosion_widths[step])  # 获取当前步骤的腐蚀宽度
        corrosion_width = max(corrosion_width, 1)  # 确保腐蚀宽度不为零

        # 在每一步中，腐蚀区域从边缘扩展
        for y in range(height):
            for x in range(width):
                if edges[y, x] > 0:  # 如果该点是边缘的一部分
                    # 计算该点是否在腐蚀区域内
                    if (x - width // 2) ** 2 + (y - height // 2) ** 2 < corrosion_width ** 2:
                        # 将腐蚀区域设为透明（模拟损坏）
                        draw = ImageDraw.Draw(pil_corrupted_image)
                        draw.ellipse([x - corrosion_width // 2, y - corrosion_width // 2,
                                       x + corrosion_width // 2, y + corrosion_width // 2],
                                     fill=(0, 0, 0, 0))  # 将腐蚀区域设为透明

        # 转回为 numpy 数组
        corrupted_image = np.array(pil_corrupted_image)

        # 将每个阶段的图像加入到列表
        corrosion_images.append(corrupted_image)

    # 计算每行的图片数（每10步为一行）
    rows = (num_steps + 9) // 10  # 计算需要多少行
    cols = 10  # 每行10个图像

    # 展示每个腐蚀阶段的图像
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.ravel()  # 将axes展平为1D数组，方便索引

    # 填充每一行的图像
    for i in range(num_steps):
        axes[i].imshow(corrosion_images[i])
        axes[i].axis('off')
        axes[i].set_title(f"Step {i + 1}")

    # 隐藏多余的子图
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 示例：调用函数并生成从边缘腐蚀的图像过程
image_path = '1.png'  # 替换为你的图像路径
simulate_corrosion_from_edge(image_path, num_steps=50, max_corrosion_size=50)
