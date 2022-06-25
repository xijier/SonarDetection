import argparse
import sys
import os
import random
import math
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets

import args
from network import StyleBankNet
import utils
from cnn_vis.cnn_visualization import get_filter, get_featuremap, vis_filter, vis_featuremap, gmm
from torchvision import utils as vutils
import cv2

SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = args.device
if torch.cuda.is_available():
    MODEL_PATH = 'model.pth'
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
else:
    MODEL_PATH = 'model_cpu.pth'
# MODEL_PATH = ''


def load_model():
    #################################
    # Load Dataset
    #################################
    style_dataset = datasets.ImageFolder(root=args.STYLE_IMG_DIR, transform=utils.img_transform)
    style_dataset = torch.cat([img[0].unsqueeze(0) for img in style_dataset], dim=0)
    style_dataset = style_dataset.to(device)
    print('dataloader done.')

    #################################
    # Define Model and Loss network (vgg16)
    #################################
    model = StyleBankNet(len(style_dataset)).to(device)

    if os.path.exists(args.GLOBAL_STEP_PATH):
        with open(args.GLOBAL_STEP_PATH, 'r') as f:
            global_step = int(f.read())
    else:
        raise Exception('cannot find global step file')
        # global_step = args.MAX_ITERATION
    if os.path.exists(args.MODEL_WEIGHT_PATH):
        model.load_state_dict(torch.load(os.path.join(args.MODEL_WEIGHT_DIR, MODEL_PATH)))
    else:
        raise Exception('cannot find model weights')
    print('network done. ({}, {}) '.format(args.MODEL_WEIGHT_DIR, global_step))
    model.eval()

    # # save as cpu version
    # m = deepcopy(model.cpu())
    # torch.save(m.state_dict(), os.path.join(args.MODEL_WEIGHT_DIR, 'model_cpu.pth'))

    return model

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    #assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)

def stylized(img, model, save_path, style_id=[0]):
    test_imgs = img.to(device)
    data = torch.zeros([args.batch_size, 3, args.IMG_SIZE, args.IMG_SIZE])
    output_test = model(test_imgs.expand_as(data), style_id)[0].cpu().detach()
    save_image_tensor2cv2(output_test, save_path)
    print('stylize finished')

if __name__ == '__main__':
    orig_model = load_model()


    # load image
    imgs_path = 'E:\kg\data\sonar_val/valinternet/shipcutoff/'
    save_path = 'E:\kg\data\sonar_val/valinternet/shipcutoff_stylebank_output/'

    files_img = os.listdir(imgs_path)
    for image_file in files_img:
        #print(image_file)
        img_path = imgs_path+image_file
        original_image = Image.open(img_path).convert('RGB')
        prep_img = utils.preprocess_image(original_image)
        stylized(prep_img, orig_model, save_path+image_file, [0])
