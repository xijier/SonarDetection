from PIL import Image
import torchvision.transforms as transforms
import torch
import lpips

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image.unsqueeze(0)  # 增加批次维度

#初始化 LPIPS 模型
lpips_model = lpips.LPIPS(net='squeeze')  # 可以选择 'alex', 'vgg', 或 'squeeze'

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])

# 加载图像
image1 = load_image('img_31_gray.jpg', transform)
image2 = load_image('img_32_gray.jpg', transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='squeeze')
lpips_fn.cuda()
x1 = torch.rand(8, 1, 128, 128).cuda()
x2 = torch.rand(8, 1, 128, 128).cuda()
loss = lpips_fn.forward(x1,x2)
print(loss)
# 计算 LPIPS 相似度
distance = lpips_model(image1, image2)

print(f'LPIPS 相似度: {distance.item()}')
