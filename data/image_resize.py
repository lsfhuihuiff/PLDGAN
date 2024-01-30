import os
import cv2
import numpy as np

image_size1 = 1080
image_size2 = 1350  #设定尺寸

source_path1 = "../datasets-old/image/"  #源文件路径
target_path1 = "../datasets-old/train_A/"  #输出目标文件路径

source_path2 = "../datasets-old/xiufu_mask/"
target_path2 = "../datasets-old/train_mask/"

image_list1 = os.listdir(source_path1)  #获得文件名
image_list2 = os.listdir(source_path2)

#i = 0
#for file in image_list1:
 #   i = i + 1
  #  if i % 100 == 0:
   #     print(file)
    #image_source = cv2.imread(source_path1 + file)  #读取图片
    #if i % 100 == 0:
     #   print(image_source.shape)
    #image = cv2.resize(image_source, (image_size1, image_size2), 0, 0,
     #                  cv2.INTER_LINEAR)  #修改尺寸
   # if i % 100 == 0:
    #    print(image.shape)
   # cv2.imwrite(target_path1 + file, image)  #重命名并且保存

print("批量处理A完成")

i = 0
for file in image_list2:
    i = i + 1
    if i % 100 == 0:
        print(file)
    image_source = cv2.imread(source_path2 + file, cv2.IMREAD_UNCHANGED)  #读取图片
    if i % 100 == 0:
        print(image_source.shape)
    print(image_source.shape)
    image = cv2.resize(image_source, (image_size1, image_size2), 0, 0,
                       cv2.INTER_LINEAR)
   # image = image[:, :, 3]   #修改尺寸
   # image[np.nonzero(image)] = 230
    #image = np.expand_dims(image, 2)
    if i % 100 == 0:
        print(image.shape)
    cv2.imwrite(target_path2 + file, image)  #重命名并且保存
print("批量处理B完成")
