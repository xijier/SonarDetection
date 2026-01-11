import cv2
import os
import pandas as pd

# # 读取彩色图像
# image = cv2.imread("img_32.jpg")
#
# # 将图像转换为灰度图像
# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# #cv2.imshow("gray",grayscale_image)
# #cv2.waitKey(0)
# # 保存灰度图像
# cv2.imwrite("img_32_gray.jpg", grayscale_image)

# import pandas as pd
#
# # 创建一个示例DataFrame
# df = pd.DataFrame({
#     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'Age': [23, 45, 25, 37]
# })
#
# # 按照年龄升序排序
# sorted_df = df.sort_values(by='Age', ascending=False)
#
# print(sorted_df)
# print(sorted_df.iloc[0])

# folder_path = "./airplane"
# folder_path_save = "./airplane_color"
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         # 获取文件的完整路径
#         full_path = os.path.join(root, file)
#         # 将文件路径添加到列表中
#         img = cv2.imread(full_path)
#         #cv2.imshow("1",img)
#         #cv2.waitKey(0)
#
#         new_width = 256
#         new_height = 256
#
#         # 调整图片大小
#         resized_image = cv2.resize(img, (new_width, new_height))
#         #grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#         # 保存调整大小后的图片
#         cv2.imwrite(os.path.join(folder_path_save, file), resized_image)
        #files_data.append({"FileName": file, "FilePath": full_path})

# 指定CSV文件的列名
# fieldnames = ["FileName", "FilePath"]

# # 准备数据
# data1 = {
#     "Name": ["Alice", "Bob", "Charlie"],
#     "Age": [28, 24, 30],
#     "City": ["New York", "Los Angeles", "Chicago"]
# }
#
# data2 = {
#     "Product": ["Widget", "Gadget", "Thingamajig"],
#     "Price": [19.99, 23.99, 12.99],
#     "Stock": [100, 150, 200]
# }
#
# # 创建DataFrame
# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)
#
# # 指定Excel文件路径
# file_path = 'output.xlsx'
# df_list = []
#
# df_list.append({"sheet_name": "Sheet_1", "Data": df1})
# df_list.append({"sheet_name": "Sheet_2", "Data": df2})
#
#
# # 将DataFrame写入Excel文件的不同工作表
# with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#     # df1.to_excel(writer, sheet_name='Sheet1', index=False)
#     # df2.to_excel(writer, sheet_name='Sheet2', index=False)
#     for df in df_list:
#         sheet_name = df["sheet_name"]
#         df1 = df["Data"]
#         df1.to_excel(writer, sheet_name=sheet_name, index=False)
#         #df2.to_excel(writer, sheet_name='Sheet2', index=False)
#
# print(f"Data has been written to {file_path}")


# import pandas as pd
# import numpy as np
# # 读取Excel文件
# file_path = 'output_color.xlsx'
# excel_file = pd.ExcelFile(file_path)
#
# score_list = []
# # 遍历每个工作表
# for sheet_name in excel_file.sheet_names:
#     # 读取工作表数据
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     # 获取第二行第二列的值 (这里的索引是从0开始的，所以第二行是索引1，第二列是索引1)
#     if len(df) > 1 and len(df.columns) > 1:
#         value = df.iloc[0]  # 使用iat访问特定位置的值
#         print(f"Value in sheet '{sheet_name}' at (1, 2): {value['Similarity']}")
#         score_list.append(value['Similarity'])
#     else:
#         print(f"Sheet '{sheet_name}' does not have enough data.")
# scores_array = np.array(score_list)
# mean_value = np.max(scores_array)
# print(mean_value)

folder_path_real = "./airplane2/"
i  = 1
for root_org, dirs_org, files_org in os.walk(folder_path_real):
    for root_org, dirs_org, files_org in os.walk(folder_path_real):
        for file_org in files_org:
            # 获取文件的完整路径
            full_path_org = os.path.join(root_org, file_org)
            # 将文件路径添加到列表中
            image = cv2.imread(full_path_org)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("./airplane3/"+str(i)+".jpg", grayscale_image)
            i+= 1
            # cv2.imshow("gray",grayscale_image)
            # cv2.waitKey(0)
# folder_path_real = "./airplane_color/airplane_08.jpg"
# image = cv2.imread(folder_path_real)
# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("airplane_08_gray.jpg", grayscale_image)

