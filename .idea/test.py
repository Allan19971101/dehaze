import torch
import torchvision
import dehazeNet
import numpy as np
from PIL import Image
import glob


def dehaze_image(image_path):
    data_hazy = Image.open(image_path)                                      #载入去雾图像
    data_hazy = (np.asarray(data_hazy) / 255.0)                             #归一化处理

    data_hazy = torch.from_numpy(data_hazy).float()                         #转换图像数据至torch类型
    data_hazy = data_hazy.permute(2, 0, 1)                                  #转换图像维度
    data_hazy = data_hazy.cuda().unsqueeze(0)                               #在第一层增加维度

    dehaze_net = dehazeNet.dehaze_net().cuda()                              #设置去雾网络
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))      #加载保存的网络

    clean_image = dehaze_net(data_hazy)                                     #得到去雾图像
    torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])   #拼接两图像，保存图像


if __name__ == '__main__':

    test_list = glob.glob("test_images/*")                #遍历测试集图片目录

    for image in test_list:
        dehaze_image(image)                                 #用训练后的网络进行去雾处理
        print(image, "done!")
