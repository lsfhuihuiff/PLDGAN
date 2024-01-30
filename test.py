"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import cv2
import numpy as np


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')
    #print(data_i['label'].size())
    img_path = data_i['path']
   # print(generated[0].shape)
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()


source_path2 = './datasets/results/sketch/test_120/images/synthesized_image/'
source_path1 = './datasets/test_A/' 
img_list1 = os.listdir(source_path1)
img_list2 = os.listdir(source_path2)

final_path = os.path.join('./datasets/results/', 'final')
if not os.path.exists(final_path):
    os.makedirs(final_path)

for file in img_list1:

    img1 = cv2.imread(source_path1+file)
    img2 = cv2.imread(source_path2+file[:-3]+'png')
    ratio = img1.shape[0] / img1.shape[1]
    print(file) 
    if img1.shape[0]>img1.shape[1]:
        img2 = cv2.resize(img2, (img1.shape[0],img1.shape[0]))
        img2 = img2[:,img2.shape[1]-img1.shape[1]:, :] 
    else:
        img2 = cv2.resize(img2, (img1.shape[1],img1.shape[1]))
        img2 = img2[img2.shape[0]-img1.shape[0]:, :, :]

    img = img2 - 0.3 * img1
    img[img<15] = 0
    img = img.astype(np.uint8)
    gray_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mask[gray_mask<50] = 0
    img = 255 - gray_mask * 1.5
    img[:, :50] = 255

    cv2.imwrite(os.path.join(final_path, file[:-3]+'png'), img)
