import cv2
import matplotlib.pyplot as plt  # plt 用于显示图片

content_path = "image/t4.png"
style_path = "image/background.png"
plt.subplot(121)  # 1行两列,第一个
figure = cv2.imread(content_path)
# 这里需要指定利用 cv 的调色板，否则 plt 展示出来会有色差
plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))

plt.subplot(122)  # 1行两列，第二个
figure = cv2.imread(style_path)
# 这里需要指定利用 cv 的调色板，否则 plt 展示出来会有色差
plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))

import PIL.Image as Image
import torchvision.transforms as transforms
img_size = 512

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    # 为img增加一个维度：1
    # 因为神经网络的输入为 4 维
    img = img.unsqueeze(0)
    return img

import torch
from torch.autograd import Variable
# 判断环境是否支持GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载风格图片
style_img = load_img(style_path)
# 对img进行转换为 Variable 对象，使它能够动态计算梯度
style_img = Variable(style_img).to(device)
# 加载内容图片
content_img = load_img(content_path)
content_img = Variable(content_img).to(device)
print(style_img.size(), content_img.size())

import torch.nn as nn


class Content_Loss(nn.Module):
    # 其中 target 表示 C ，input 表示 G，weight 表示 alpha 的平方
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        # detach 可以理解为使 target 能够动态计算梯度
        # target 表示目标内容，即想变成的内容
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss
# 损失函数的测试
cl = Content_Loss(content_img, 1)
# 随机图片
rand_img = torch.randn(content_img.data.size(), device=device)
cl.forward(rand_img)
print(cl.loss)

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        # 将特征图变换为 2 维向量
        feature = input.view(a * b, c * d)
        # 内积的计算方法其实就是特征图乘以它的逆
        gram = torch.mm(feature, feature.t())
        # 对得到的结果取平均值
        gram /= (a * b * c * d)
        return gram

gram = Gram()
gram

target = gram(style_img)
# 此时 style_img 的通道为3 所以产生的风格特征为 3×3
target

class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        # weight 和内容函数相似，表示的是权重 beta
        self.weight = weight
        # targer 表示图层目标。即新图像想要拥有的风格
        # 即保存目标风格
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        # 加权计算 input 的 Gram 矩阵
        G = self.gram(input) * self.weight
        # 计算真实的风格和想要得到的风格之间的风格损失
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out
    # 向后传播

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss

# 传入模型所需参数
sl = Style_Loss(target, 1000)
# 传入一张随机图片进行测试
rand_img = torch.randn(style_img.data.size(), device=device)
# 损失函数层向前传播，进而得到损失
sl.forward(rand_img)
sl.loss

import torchvision.models as models
# 设置与预训练模型所在连接
#torch.utils.model_zoo.load_url("https://labfile.oss.aliyuncs.com/courses/861/vgg19_pre.zip")

#cnn = models.vgg19(pretrained=True).features.to(device).eval()

#pretrained_model = models.vgg19(pretrained=True).features
pretrained_model = models.vgg19(pretrained=True).features
pretrained_model = pretrained_model.to(device)
pretrained_model

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# 初始化一个 空的神经网络 model
model = nn.Sequential()
model = model.to(device)
# 构造网络模型，并且返回这些损失函数


def get_style_model_and_loss(style_img, content_img, cnn=pretrained_model, style_weight=1000, content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    # 用列表来存上面6个损失函数
    content_loss_list = []
    style_loss_list = []

    # 风格提取函数
    gram = Gram()
    gram = gram.to(device)

    i = 1
    # 遍历 模型（vgg19,mobilenet, etc） ，找到其中我们需要的卷积层
    for layer in cnn:
        # 如果 layer 是  nn.Conv2d 对象，则返回 True
        # 否则返回 False
        if isinstance(layer, nn.Conv2d):
            # 将该卷积层加入我们的模型中
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            # 判断该卷积层是否用于计算内容损失
            if name in content_layers_default:
                # 这里是把目标放入模型中，得到该层的目标
                target = model(content_img)
                # 目标作为参数传入具体的损失类中，得到一个工具函数。
                # 该函数可以计算任何图片与目标的内容损失
                content_loss = Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            # 和内容损失相似，不过增加了一步：提取风格
            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                # 目标作为参数传入具体的损失类中，得到一个工具函数。
                # 该函数可以计算任何图片与目标的风格损失
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        # 对于池化层和 Relu 层我们直接添加即可
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)
    # 综上：我们得到了：
    # 一个具体的神经网络模型，
    # 一个风格损失函数集合（其中包含了 5 个不同风格目标的损失函数）
    # 一个内容损失函数集合（这里只有一个，你也可以多定义几个）
    return model, style_loss_list, content_loss_list

model, style_loss_list, content_loss_list = get_style_model_and_loss(
    style_img, content_img)
model

import torch.optim as optim


def get_input_param_optimier(input_img):
    # 将input_img的值转为神经网络中的参数类型
    input_param = nn.Parameter(input_img.data)
    # 告诉优化器，我们优化的是 input_img 而不是网络层的权重
    # 采用 LBFGS 优化器
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


# 输入一个随机图片进行测试
get_input_param_optimier(rand_img)

# 传入的 input_img 是 G 中每个像素点的值，可以为一个随机图片
def run_style_transfer(content_img, style_img, input_img, num_epoches):
    print('Building the style transfer model..')
    # 指定所需要优化的参数，这里 input_param就是G中的每个像素点的值
    input_param, optimizer = get_input_param_optimier(input_img)

    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:
        # 这里我们自定义了总损失的计算方法
        def closure():
            input_param.data.clamp_(0, 1)  # 更新图像的数据
            # 将此时的 G 传入模型中，得到每一个网络层的输出
            model(input_param)
            style_score = 0
            content_score = 0
            # 清空之前的梯度
            optimizer.zero_grad()
            # 计算总损失，并得到各个损失的梯度
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            # 这里每迭代一次就进行一次输出
            # 你可以根据自身情况进行调节
            if epoch[0] % 1 == 0:
                print('run {}/80'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data.item(), content_score.data.item()))
                print()

            return style_score + content_score
        # 更新 G
        optimizer.step(closure)
    # 返回训练完成的 G，此时的 G
    return input_param.data

# 初始化 G
input_img = content_img.clone()
# 进行模型训练，并且返回图片
out = run_style_transfer(content_img, style_img, input_img, num_epoches=50)
# 将图片转换成可 PIL 类型，便于展示
new_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
print("训练完成")

# 展示图片
plt.imshow(new_pic)
plt.show()