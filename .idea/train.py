import torch
import torch.nn as nn
import torchvision
import os
import argparse
import dataloader
import dehazeNet
import numpy as np


def weights_init(m):                                                          #权重初始化
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    dehaze_net = dehazeNet.dehaze_net().cuda()                                 #网络初始化
    dehaze_net.apply(weights_init)                                             #参数初始化

    train_dataset = dataloader.dehazing_loader(config.orig_images_path,        #加载训练集测试集
                                               config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()                                            #选用均方损失函数
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)      #采用默认参数，0.0001学习率

    dehaze_net.train()

    for epoch in range(config.num_epochs):                                    #迭代次数
        for iteration, (img_orig, img_haze) in enumerate(train_loader):       #枚举训练集
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)                                 #训练网络

            loss = criterion(clean_image, img_orig)                            #定义均方差为损失值

            optimizer.zero_grad()                                              #每次迭代梯度置0
            loss.backward()                                                    #回馈函数
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)           #梯度剪裁，设置最大梯度为0.1
            optimizer.step()                                                   #更新训练模型

            if ((iteration + 1) % config.display_iter) == 0:                  #每10步输出一次均方差
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:                 #每200步保存一次当前模型
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):          #枚举测试集
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)                                 #进行利用训练后的算法去雾

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),               #保存去雾前和去雾后的拼接图片
                                         config.sample_output_folder + str(iter_val + 1) + ".png")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")                  #保存最终训练后的模型


if __name__ == "__main__":

    parser = argparse.ArgumentParser()                      #命令行参数设置

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="D:\\新建文件夹\\ITS_v2\\clear\\")
    parser.add_argument('--hazy_images_path', type=str, default="D:\\新建文件夹\\ITS_v2\\hazy\\hazy\\")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="C:\\Users\\lzq\\PycharmProjects\\pytorch_dehaze\\.idea\\snapshots\\")
    parser.add_argument('--sample_output_folder', type=str, default="C:\\Users\\lzq\\PycharmProjects\\pytorch_dehaze\\.idea\\samples\\")

    config = parser.parse_args()                            #默认参数生效

    if not os.path.exists(config.snapshots_folder):        #检查临时模型文件夹和样本文件夹
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)                                           #按照默认参数训练




