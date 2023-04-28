# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from seaborn import color_palette
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
from utils import *
import math
import sys
# %matplotlib inline

# # CONVERT IMAGE TO TENSOR

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 归一化到[0, 1] /255 HWC->C*H*W
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_path = list(template_dir_path.iterdir())
        self.image_name = image_name
        
        self.image_raw = cv2.imread(self.image_name)
        
        self.thresh_df = None
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)

        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)

        """self.arrs = []
        for path in self.template_path:
            template = cv2.imread(str(path))
            self.arrs.append(self.kkk(template))"""

    def kkk(self, template):
        ht, wt, _ = template.shape
        tgray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        tgray = np.array(tgray)
        white_num = np.where(tgray == 255)[0].shape[0]
        th_arr = 1 - white_num / (ht * wt)
        # print(th_arr)

        hi, wi, _ = self.image_raw.shape
        igray = cv2.cvtColor(self.image_raw, cv2.COLOR_BGR2GRAY)

        igray = np.array(igray)
        # print(igray.shape)
        arrs = igray
        arrs[:, :wt // 2] = 1
        arrs[:, math.ceil(-wt / 2):] = 1  # math.ceil向上舍入到最接近的整数
        arrs[:ht // 2, :] = 1
        arrs[math.ceil(-ht / 2):, :] = 1

        kernel = np.ones((ht, wt))
        ret, imask = cv2.threshold(igray, 0, 1, cv2.THRESH_BINARY)
        img_bool = np.array(imask).astype("bool")
        img_bool = ~ img_bool
        img_int01 = img_bool.astype(np.int)
        black_sums = cv2.filter2D(img_int01, -1, kernel)  # 用template大小的核进行均值滤波， -1 保证输入输出相同尺寸

        r = (1 - black_sums[ht // 2: hi + math.ceil(-ht / 2), wt // 2: wi + math.ceil(-wt / 2)] / (ht * wt))/th_arr
        arrs[ht // 2: hi + math.ceil(-ht / 2), wt // 2: wi + math.ceil(-wt / 2)] = r
        # print(arrs.max())
        arrs = np.where((arrs < 1.1) & (arrs > 0.9), 1, 0.001)
        # arrs = np.exp(-arrs + 1)
        # print(p1, p2)

        """p1 = 0
        p2 = 0
        for i in range(wt // 2, wi + math.ceil(-wt / 2)):
            for j in range(ht // 2, hi + math.ceil(-ht / 2)):
                black_num = \
                    np.where(igray[j - ht // 2:j + math.ceil(ht / 2), i - wt // 2: i + math.ceil(wt / 2)] == 0)[
                        0].shape[0]
                if j <= ht // 2 + 2 :
                    print(black_num)
                else:
                    sys.exit()
                r = (1 - black_num / (ht * wt))/th_arr
                # print(j, i, r)
                if r > 1.5:
                    arrs[j, i] = np.exp(-r)
                    p1 += 1
                else:
                    arrs[j, i] = 1
                    p2 += 1
        print(p1, p2)"""
        return arrs
        
    def __len__(self):
        return len(self.template_names)

    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template = cv2.imread(template_path)
        image_r = self.kkk(template)
        if self.transform:
            template = self.transform(template)
        thresh = 0.7
        if self.thresh_df is not None:
            if self.thresh_df.path.isin([template_path]).sum() > 0:
                thresh = float(self.thresh_df[self.thresh_df.path==template_path].thresh)
        return {'image': self.image, 
                'image_raw': self.image_raw,
                'image_name': self.image_name,
                'template': template.unsqueeze(0),
                'template_name': template_path,
                'template_h': template.size()[-2],
                'template_w': template.size()[-1],
                'thresh': thresh,
                'image_r': image_r}




# template_dir = 'template/'
# image_path = 'sample/sample1.jpg'
# dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')


# ### EXTRACT FEATURE

class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)
        
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        
    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        if mode=='big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        else:        
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)


class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _ , H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]


class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None
    def __call__(self, template, image, image_name):
        T_feat = self.featex(template)
        if self.I_feat_name is not image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True), T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))  # 爱因斯坦求和约定
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps


class QATM():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()  # x a b c d
        x = x.view(batch_size, ref_row*ref_col, qry_row*qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha*xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row*ref_col))  # 生成网格
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values
    def compute_output_shape( self, input_shape ):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)


# # NMS AND PLOT

# ## SINGLE

def nms(score, w_ini, h_ini, thresh=0.7):
    dots = np.array(np.where(score > thresh*score.max()))
    
    x1 = dots[1] - w_ini//2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini//2
    y2 = y1 + h_ini

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.5)[0]
        order = order[inds + 1]
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes


def plot_result(image_raw, boxes, show=False, save_name=None, color=(255, 0, 0)):
    # plot result
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 3)
    if show:
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img


# ## MULTI

def nms_multi(scores, w_array, h_array, thresh_list):
    indices = np.arange(scores.shape[0])  # template索引--t scores.shape = (t, h, w)
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)  # 取每个template匹配search图片的score最大值
    # omit not-matching templates
    scores_omit = scores[maxes > 0.1 * maxes.max()]  # 取大于最大值90%的template (t, h, w)
    # print("scores_omit:{}".format(scores_omit.max()))
    indices_omit = indices[maxes > 0.1 * maxes.max()]
    # print("indices_omit:{}".format(indices_omit.shape))
    # extract candidate pixels from scores
    dots = None
    dos_indices = None
    for index, score in zip(indices_omit, scores_omit):  # 对每个t和s
        dot = np.array(np.where(score > thresh_list[index]*score.max()))  # s图片按阈值mask, np.where返回坐标元组
        # print("dot: {}".format(dot.shape))  # (2, x)
        if dots is None:  # 初始化
            # print("N")
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    # print(dots)
    # print(dots_indices)
    dots_indices = dots_indices.astype(np.int)
    x1 = dots[1] - w_array[dots_indices]//2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices]//2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    max_score = scores[order][0]  # for analysis.py
    mean_score = scores.mean()
    shape = scores.shape
    # print("order:{}".format(order))
    # print("scores:{}".format(scores))
    # print("sccores[order]:{}".format(scores[order]))
    dots_indices = dots_indices[order]
    # print("dots_indices[order]:{}".format(dots_indices))
    
    keep = []
    keep_index = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
        
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2,0,1)
    return boxes, np.array(keep_index), max_score, mean_score, shape


def plot_result_multi(image_raw, boxes, indices, show=False, save_name=None, color_list=None):
    d_img = image_raw.copy()
    if color_list is None:
        color_list = color_palette("hls", indices.max()+1)
        color_list = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), color_list))
    for i in range(len(indices)):
        d_img = plot_result(d_img, boxes[i][None, :,:].copy(), color=color_list[indices[i]])
    if show:
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img


# # RUNNING

def run_one_sample(model, template, image, image_name, image_r):
    val = model(template, image, image_name)  # [batch_size, ref_row, ref_col, 1]
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)
    
    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i,:,:,0]
        # print(gray)
        gray = cv2.resize( gray, (image.size()[-1], image.size()[-2]) )
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score(gray, w, h)
        # print(gray.min())
        # print(score.shape)
        score[score>-1e-7] = score.min()

        # print(score.max())
        score = np.exp(score / (h*w))
        # score = np.multiply(score, image_r)  # 面积惩罚系数
        # score -= image_r  # 面积惩罚常数
        # reverse number range back after computing geometry average 在计算几何平均数后反转数字范围
        scores.append(score)
    return np.array(scores)


def run_multi_sample(model, dataset):
    scores = []
    w_array = []
    h_array = []
    thresh_list = []
    for data in dataset:  # 遍历template集合
        # print("-----------------")
        score = run_one_sample(model, data['template'], data['image'], data['image_name'], data['image_r'])
        scores.append(score)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.squeeze(np.array(scores), axis=1), np.array(w_array), np.array(h_array), thresh_list


"""model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=True)

scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)

boxes, indices, _, _ = nms_multi(scores, w_array, h_array, thresh_list)

d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result_sample.png')

plt.imshow(scores[2])
"""
