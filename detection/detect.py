import numpy as np
import torch
import os
from detection.unet_resnet import UNetResNet
import torch.utils.data as data
# import torchvision.transforms as transforms
from albumentations import *
# from imutils import *
import cv2
from tqdm import tqdm


# from matplotlib import pyplot as plt


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def perpectiveTransform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


class Cars(data.Dataset):
    def __init__(self, jsonfile, img_folder, testimg, out_res=(256, 512), train=True):
        self.img_folder = img_folder  # root image folders
        self.out_res = out_res
        self.train = train

        # if train:
        # self.anno = os.listdir('data/train/')
        # else:
        #     self.anno = os.listdir('data/test/')

        self.anno = [testimg]

        self.transform = Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __getitem__(self, index):
        file_name = self.anno[index]

        # img_path = os.path.join(self.img_folder, file_name)
        img_path = os.path.join(os.getcwd(), file_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.out_res[1], self.out_res[0]))

        aug = self.transform(image=img)
        inp = aug['image']
        inp = torch.from_numpy(np.transpose(inp, (2, 0, 1)).astype('float32'))

        meta = {'original_shape': img.shape, 'img_name': img_path}

        return inp, img, meta

    def __len__(self):
        return len(self.anno)


def detect(testimg, model="checkpoint/model_best.pth.tar"):
    weights = torch.load(model, map_location=lambda storage, loc: storage)
    start_epoch = weights['epoch']
    best_acc = weights['best_acc']

    keys = weights['state_dict'].keys()
    new_dict = {}
    for old_key in keys:
        new_key = '.'.join(old_key.split('.')[1:])
        new_dict[new_key] = weights['state_dict'][old_key]

    model = UNetResNet(num_classes=4)
    model.load_state_dict(new_dict)

    # print(model)

    model.to('cpu')

    test_batch = 1
    workers = 1

    loader = torch.utils.data.DataLoader(
        Cars('data/cars_annotations.json', 'data/', testimg=testimg, train=False),
        batch_size=test_batch, shuffle=False,
        num_workers=workers, pin_memory=True)

    # print(len(loader))
    # exit(0)

    for i, (inputs, orig, meta) in tqdm(enumerate(loader)):
        # print(i)
        # print(meta)

        img = cv2.imread(meta["img_name"][0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 256))

        input_var = torch.autograd.Variable(inputs.cpu(), volatile=True)
        output = model(input_var)
        score_map = output.data.cpu()

        # print(score_map)
        # exit(0)

        top_left = np.unravel_index(score_map[0][0].argmax(), score_map[0][0].shape)[::-1]
        bottom_left = np.unravel_index(score_map[0][1].argmax(), score_map[0][1].shape)[::-1]
        bottom_right = np.unravel_index(score_map[0][2].argmax(), score_map[0][2].shape)[::-1]
        top_right = np.unravel_index(score_map[0][2].argmax(), score_map[0][2].shape)[::-1]

        top_right = list(top_right)
        top_right[0] += abs(top_left[0] - bottom_left[0])
        top_right[1] -= abs(top_left[1] - bottom_left[1])
        top_right = tuple(top_right)

        # print(top_left)
        # print(top_right)
        # print(bottom_right)
        # print(bottom_left)

        # cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

        pts = np.array([[top_left, top_right, bottom_right, bottom_left]], np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, (0, 255, 255))

        mask = np.zeros(img.shape, dtype=np.uint8)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, pts, ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex

        # apply the mask
        masked_image = cv2.bitwise_and(img, mask)

        warped = perpectiveTransform(img, pts[0])

        return warped

        # cv2.imshow("warped", warped)
        # cv2.waitKey(0)
        # cv2.imwrite("warped" + str(i) + ".jpg", warped)

        # cv2.imshow("masked_image", masked_image)
        # cv2.waitKey(0)

        # print("top_left", top_left)
        # print("bottom_right", bottom_right)

        # pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        # crop_img = masked_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] #img[y:y + h, x:x + w]
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        # cv2.imwrite("cropped" + str(i) + ".jpg", crop_img)
        # cv2.waitKey(0)

        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # exit(0)
        # print()

#
# img = detect("../ford.png")
#
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
