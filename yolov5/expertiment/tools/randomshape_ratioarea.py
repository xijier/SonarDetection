import math

import matplotlib.pyplot as plt
from random import choice  # random是系統自帶的隨機函式模組
import cv2
import itertools
import random
import numpy as np

class Rand_moving():  # 定義一個Rand_moving類
    def __init__(self, num_times=1000):  # 初始化，設定預設引數為10萬，可以修改這個引數試試機器執行速度
        self.num_times = num_times  # 移動次數

        self.x_values = [0]  # 設定兩個數列，用來儲存每一步的位置，初始位置為(0, 0)，數列元素個數會一直增加到num_times，用來記錄每一步的位置資訊
        self.y_values = [0]

    def fill_moving(self):  # 定義一個函式，用來計算移動方向和距離，並計算需要儲存的位置資訊
        xy_values = []
        formatList = []
        x_values_a = []
        y_values_a = []
        x_values_a.append(0)
        y_values_a.append(0)
        xy_values.append((0, 0))
        formatList.append((0, 0))
        #while len(self.x_values) < self.num_times or len(formatList) < self.num_times -1:  # 迴圈不斷執行，直到漫步包含所需數量的點num_times
        while len(self.x_values) < self.num_times:
            x_direction = choice([1, -1])  # x的移動方向，1向上，0不變，-1向下
            x_distance = choice([0, 1, 2, 3, 4, 5])  # x的每次移動的畫素，
            x_step = x_direction * x_distance  # 移動方向乘以移動距離，以確定沿x移動的距離

            y_direction = choice([1, -1])  # y的移動方向，1向上，0不變，-1向下
            y_distance = choice([0, 1, 2, 3, 4, 5])  # y的每次移動的畫素，
            y_step = y_direction * y_distance  # 移動方向乘以移動距離，以確定沿y移動的距離

            # 原地不變
            if x_step == 0 and y_step == 0:  # x_step和 y_step都為零，則意味著原地踏步
                continue

            # 計算下一個點的位置座標x和y值，並分別儲存到數列x_values和y_values中
            next_x = self.x_values[-1] + x_step  # self.x_values[-1]表示是數列最後一個值，初始為x_values=[0]
            next_y = self.y_values[-1] + y_step
            while next_x > 256 or next_x < -256:
                x_direction = choice([1, -1])  # x的移動方向，1向上，0不變，-1向下
                x_distance = choice([0, 1, 2, 3, 4, 5])  # x的每次移動的畫素，
                x_step = x_direction * x_distance  # 移動方向乘以移動距離，以確定沿x移動的距離
                next_x = self.x_values[-1] + x_step

            while next_y > 256 or next_y < -256:
                y_direction = choice([1, -1])  # y的移動方向，1向上，0不變，-1向下
                y_distance = choice([0, 1, 2, 3, 4, 5])  # y的每次移動的畫素，
                y_step = y_direction * y_distance  # 移動方向乘以移動距離，以確定沿y移動的距離
                next_y = self.y_values[-1] + y_step

            xy_values.append((next_x,next_y))
            #formatList = list(set(xy_values))
            self.x_values.append(next_x)  # 將每次計算的next_x存入到數列x_values中
            self.y_values.append(next_y)  # 將每次計算的next_y存入到數列y_values中

            # max_value_x = max(self.x_values)
            # min_value_x = min(self.x_values)
            # max_value_y = max(self.y_values)
            # min_value_y = min(self.y_values)
            # x = max_value_x - min_value_x
            # y = max_value_y - min_value_y
            # total_area = x*y

        #b = list(zip(*formatList))
        #self.x_values = b[0]
        #self.y_values = b[1]
        print("complete.")

class New_case():
    # 定义New_case类
    def __init__(self, numbers):  # 定义要创建的实例个数
        self.pic_x = 512
        self.pic_y = 512
        self.area_rate = 0.25
        self.numbers = numbers
        self.caselist = {}  # 定义一个空的cases列表
        self.case = 0  # 定义一个case变量
        self.colors = {'1': 'red', '2': 'orange', '3': 'yellow', '4': 'green', '5': 'blue',
                       '6': 'puple'}  # 创建了一新字典colors{}，将生成的个数与颜色相对应

        while self.case < self.numbers:  # 小于给定实例个数时
            self.case += 1
            #times = choice([100000, 150000, 200000, 250000])  # 随机生成一个移动次数
            #times = choice([10000, 15000, 20000, 25000])  # 随机生成一个移动次数
            n = self.pic_x*self.pic_y * self.area_rate # scatter 为像素 “,”
            #n = 10 ** 2 * area_rate / (math.pi * r[0] ** 2) # scatter 为像素 “o”
            times = n
            #print(times)
            self.caselist[self.case] = times  # 将变量case作为key, times作为value保存到字典中

    def case_moving(self):  # 重新定义一个方法，即访问字典所有项
        for key, value in self.caselist.items():  # 字典不为空
            examplecase = Rand_moving(int(value))  # 创建实例，将对应的value值传递类Rand_moving
            examplecase.fill_moving()  # 调用类Rand_moving中的方法fill_moving()计算移动相关数据并保存到列表中
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, aspect='equal')
            #ax.grid(True)  # 显示网格线
            max_value_x = max(examplecase.x_values)
            min_value_x = min(examplecase.x_values)
            max_value_y = max(examplecase.y_values)
            min_value_y = min(examplecase.y_values)
            ax.set_xlim(min_value_x, max_value_x)
            ax.set_ylim(min_value_y, max_value_y)
            #ax.grid(True)
            r = [0.5]  # 半径
            rr_pix = (ax.transData.transform(np.vstack([r, r]).T) - ax.transData.transform(
                np.vstack([np.zeros(1), np.zeros(1)]).T))
            rpix, _ = rr_pix.T
            size_pt = (2 * rpix / fig.dpi * 72) ** 2  # 像素点与单个点面积换算
            plt.scatter(examplecase.x_values, examplecase.y_values, c=self.colors["5"], s=size_pt,marker="o")
            plt.axis('off')
            plt.savefig("../data/area_stain/"+str(key)+"_28.png", format='png', bbox_inches='tight', transparent=True, dpi=600)
            #plt.figure(dpi=128, figsize=(12, 10))  # 创建画面屏幕
            #plt.scatter(examplecase.x_values, examplecase.y_values, c=self.colors[colorkey], s=15)  # 注意调用了上述新字典的颜色
        plt.show()

n = 1 # 产生多少个随机图
testcase = New_case(int(n))  # 将n转为整型数据，创建实例个数
testcase.case_moving()

#
# import matplotlib.pyplot as plt
#

# x = [1, 2, 3, 4]
# y = [1, 2, 3, 4]
# n  = 4 # 点的个数
# r = [0.5]# 半径
#
# pic_x = 512
# pic_y = 512
# area_rate = 0.3
# n = 10**2 * area_rate/(math.pi*r[0]**2) # 占用30%的面积需要点个数, 100**2 为整个图像的面积
# n = pic_x*pic_y * area_rate
# print(n)
#
# random_list = list(itertools.product(range(1, pic_x), range(1, pic_y)))
# random_list = []
# for i in range(0,pic_x):
#     for j in range(0,pic_y):
#         random_list.append((i,j))
# a = random.sample(random_list, int(n))
# b = list(zip(*a))
# x_values = b[0]
# y_values = b[1]
# print(len(x_values))
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, aspect='equal')
# #ax.grid(True)  # 显示网格线
# ax.set_xlim(0, pic_x)
# ax.set_ylim(0, pic_y)
# rr_pix = (ax.transData.transform(np.vstack([r, r]).T) - ax.transData.transform(np.vstack([np.zeros(1), np.zeros(1)]).T))
# rpix, _ = rr_pix.T
# size_pt = (2 * rpix / fig.dpi * 72) ** 2 # 像素点与单个点面积换算
# scat = ax.scatter(x_values, y_values, s=size_pt, alpha=0.5,marker=",")
# print(size_pt)
# fig.canvas.draw()
# plt.show()