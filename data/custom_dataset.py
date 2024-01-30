"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from re import T
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(preprocess_mode='scale_width_and_crop')
            parser.set_defaults(crop_size=480)
        else:
            parser.set_defaults(preprocess_mode='scale_width')
            parser.set_defaults(crop_size=512)
        parser.set_defaults(load_size=512)
    
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=0)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, default='./datasets/test_A',
                            help='path to the directory that contains label images')
        parser.add_argument('--mask_dir', type=str, default='./datasets/test_mask',
                            help='path to the directory that contains mask images')
        parser.add_argument('--joints_dir', type=str, default='./datasets/test_joints_img',
                            help='path to the directory that contains joints images')
        parser.add_argument('--image_dir', type=str, default='./datasets/test_A',
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
    
        mask_dir = opt.mask_dir
        mask_paths = make_dataset(mask_dir, recursive=False, read_cache=True)

        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []
       # print(image_paths)
       # print(label_paths)
       # assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
       # assert len(label_paths) == len(mask_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths, mask_paths
