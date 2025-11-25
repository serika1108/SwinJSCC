from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data

NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class Train_image(Dataset):
    def __init__(self, args):
        self.imgs = []
        # 搜索当前目录下所有jpg和png的图像，并拼接在一起
        self.imgs += glob(os.path.join(args.train_data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(args.train_data_dir, '*.png'))
        self.crop_size = args.image_dims
        self.transform = self._transforms()

    def _transforms(self, ):
        transforms_list = [
            # 训练数据集固定分辨率
            transforms.RandomCrop((self.crop_size, self.crop_size)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Test_image(Dataset):
    def __init__(self, args):
        self.imgs = []
        self.imgs += glob(os.path.join(args.test_data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(args.test_data_dir, '*.png'))
        self.mini_size = args.window_size * args.patch_size * (2 ** (len(args.encoder_depth)-1))

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % self.mini_size != 0 or self.im_width % self.mini_size != 0:
            self.im_height = self.im_height - self.im_height % self.mini_size
            self.im_width = self.im_width - self.im_width % self.mini_size
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


def get_train_loader(args):
    train_dataset = Train_image(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    return train_loader


def get_test_loader(args):
    test_dataset = Test_image(args)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)


    return test_loader
