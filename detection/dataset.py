from __future__ import print_function, absolute_import

# import os
# import numpy as np
import json
# import random
# import math

import torch
import torch.utils.data as data
# import torchvision.transforms as transforms
from albumentations import *
from detection.imutils import *
import cv2


class Cars(data.Dataset):
    def __init__(self, jsonfile, img_folder, out_res=(256, 512), train=True):
        self.img_folder = img_folder  # root image folders
        self.is_train = train  # training set or test set
        self.out_res = out_res

        # create train/val split
        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if self.anno[idx]['Label'] != 'Skip':
                if idx < 100:
                    self.valid.append(idx)
                else:
                    self.train.append(idx)

        if self.is_train:
            self.transform = Compose([
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.6),
                HorizontalFlip(p=0.05),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.05),
                OneOf([
                    MotionBlur(p=.1),
                    MedianBlur(blur_limit=3, p=.3),
                    Blur(blur_limit=3, p=.3),
                ], p=0.6),
                OneOf([
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=.1),
                    IAAPiecewiseAffine(p=0.3),
                ], p=0.3),
                OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomContrast(),
                    RandomBrightness(),
                    RandomGamma()
                ], p=0.6),
                HueSaturationValue(p=0.7),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = Compose([
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __getitem__(self, index):
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        points = a['Label']['Licence plate'][0]['geometry']

        points.sort(key=lambda point: point['x'])
        top_left = points[0] if points[0]['y'] < points[1]['y'] else points[1]
        points.remove(top_left)
        bottom_left = points[0]
        points.remove(bottom_left)
        points.sort(key=lambda point: point['y'])
        top_right = points[0]
        points.remove(top_right)
        bottom_right = points[0]

        top_left = (top_left['x'], top_left['y'])
        bottom_left = (bottom_left['x'], bottom_left['y'])
        top_right = (top_right['x'], top_right['y'])
        bottom_right = (bottom_right['x'], bottom_right['y'])

        # top_left = #(points[0]['x'], points[0]['y'])
        # bottom_left = (points[1]['x'], points[1]['y'])
        # bottom_right = (points[2]['x'], points[2]['y'])
        # top_right = (points[3]['x'], points[3]['y'])
        points = [top_left, bottom_left, bottom_right, top_right]

        # print(points)

        img_path = os.path.join(self.img_folder, a['External ID'])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_shape = img.shape

        img = cv2.resize(img, (self.out_res[1], self.out_res[0]))

        ry = self.out_res[0] / orig_shape[0]
        rx = self.out_res[1] / orig_shape[1]

        for i in range(len(points)):
            points[i] = (points[i][0] * rx, points[i][1] * ry)

        # Generate ground truth
        target = np.zeros((self.out_res[0], self.out_res[1], len(points)))

        for i in range(len(points)):
            target[:, :, i] = generate_hm(self.out_res[1], self.out_res[0], points[i])

        augmented = self.transform(image=img, mask=target)

        inp = augmented['image']
        target = augmented['mask']

        inp = torch.from_numpy(np.transpose(inp, (2, 0, 1)).astype('float32'))
        target = torch.from_numpy(np.transpose(target, (2, 0, 1)).astype('float32'))

        # Meta info
        meta = {'index': index, 'pts': points}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
