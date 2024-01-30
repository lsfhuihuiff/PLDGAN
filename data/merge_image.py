import os
import cv2
import numpy as np

image_size1 = 1080
image_size2 = 1350  #设定尺寸
source_path1 = "../datasets/train_A/"  #源文件路径
#target_path1 = "../datasets/train_A/"  #输出目标文件路径

source_path2 = "../datasets/train_B_gray/"
target_path = "../datasets/train_B/"

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
   # cv2.imwrite(target_path1 + file, image)  #重命名并且保存

print("批量处理A完成")

i = 0
for file in image_list1:
    i = i + 1
    if i % 100 == 0:
        print(file)
    image_source1 = cv2.imread(source_path1 + file, cv2.IMREAD_UNCHANGED)  #读取>图片
    image_source2 = cv2.imread(source_path2 + file[:-3] + 'png', cv2.IMREAD_UNCHANGED)
    image_source2 = np.expand_dims(image_source2, 2).repeat(3, axis=2)
    if i % 100 == 0:
        print(image_source2.shape)
    #print(image_source2.shape)
    #print(image_source1.shape)
    image = cv2.addWeighted(image_source1, 0.2, image_source2, 0.8, 0)
    #image = image[:, :, 3]   #修改尺寸
    #image[np.nonzero(image)] = np.random.randint(200,255)
    #image = np.expand_dims(image, 2)
    if i % 100 == 0:
        print(image.shape)
    cv2.imwrite(target_path + file, image)  #重命名并且保存
print("批量处理B完成")
                                                             
