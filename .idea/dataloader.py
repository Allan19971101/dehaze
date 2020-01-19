import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(11)                                                                #设置随机数


def populate_train_list(orig_images_path, hazy_images_path):
    train_list = []                                                            #训练列表
    val_list = []                                                              #测试列表

    image_list_haze = glob.glob(hazy_images_path + "*.png")                   #加载雾图图像

    tmp_dict = {}                                                              #暂用列表

    for image in image_list_haze:                                              #遍历所有的雾图
        image = image.split("\\")[-1]                                           #图片名
        key = image.split("_")[0]  + ".png"                                   #遍历原图，每张原图对应9张雾图
        if key in tmp_dict.keys():                                             #遍历所有图片，并加载在tmp_dict里
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    train_keys = []
    val_keys = []

    len_keys = len(tmp_dict.keys())                                             #计算总图片数
    for i in range(len_keys):                                                  #90%图片用作训练集，10%用作测试集
        if i < len_keys * 9 / 10:
            train_keys.append(list(tmp_dict.keys())[i])
        else:
            val_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):                                           #存入训练集和测试集图片

        if key in train_keys:
            for hazy_image in tmp_dict[key]:
                train_list.append([orig_images_path + key, hazy_images_path + hazy_image])

        else:
            for hazy_image in tmp_dict[key]:
                val_list.append([orig_images_path + key, hazy_images_path + hazy_image])

    random.shuffle(train_list)                                                  #打乱训练集测试集分布
    random.shuffle(val_list)

    return train_list, val_list


class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):

        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)        #存入图片

        if mode == 'train':                                                #训练集列表
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list                                  #测试集列表
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_orig_path, data_hazy_path = self.data_list[index]              #路径

        data_orig = Image.open(data_orig_path)
        data_hazy = Image.open(data_hazy_path)

        data_orig = data_orig.resize((480, 640), Image.ANTIALIAS)           #图像改为480*640
        data_hazy = data_hazy.resize((480, 640), Image.ANTIALIAS)

        data_orig = (np.asarray(data_orig) / 255.0)                         #归一化
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()                     #转换成torch类型
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)       #维度变换为（3*480*640）

    def __len__(self):
        return len(self.data_list)                                          #返回列表长度

