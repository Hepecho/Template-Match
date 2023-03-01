import cv2
import numpy as np
import sys
from PIL import Image
import os
import shutil

def affine_transform(src):
    # 获取图像shape
    rows, cols = src.shape[: 2]

    # 设置图像仿射变化矩阵
    post1 = np.float32([[50, 50], [200, 50], [50, 200]])
    post2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(post1, post2)

    # 图像变换
    result = cv2.warpAffine(src, M, (rows, cols))

    # 图像显示
    cv2.imshow("result", result)
    cv2.imshow("scr", src)

    # 等待窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pers_transform(src):
    # 设置透视变换图像矩阵
    post1 = np.float32([[55, 65], [288, 49], [28, 237], [239, 240]])
    post2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
    M = cv2.getPerspectiveTransform(post1, post2)

    # 图像变换
    result = cv2.warpPerspective(src, M, (200, 200))

    # 图像显示
    cv2.imshow("scr", src)
    cv2.imshow("result", result)

    # 等待窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bg_add_fg(bg, fg, x, y):
    r0, c0 = bg.shape
    r1, c1 = fg.shape
    roi = bg[x:x + r1, y:y + c1]
    ret, mask = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)
    # ROI掩模区域反向掩模
    mask_inv = cv2.bitwise_not(mask)
    # 掩模显示背景
    # Now black-out the area of logo in ROI
    roi_ = cv2.bitwise_and(roi, roi, mask=mask)

    print('roi_')
    cv2.imshow('roi_', roi_)
    cv2.waitKey(0)

    # 掩模显示前景
    # Take only region of logo from logo image.
    src_fg = cv2.bitwise_and(src, src, mask=mask_inv)
    print('src_fg')
    cv2.imshow('src_fg', src_fg)
    cv2.waitKey(0)

    # 前背景图像叠加
    # Put logo in ROI and modify the main image
    dst = cv2.add(roi_, src_fg)
    bg[x:x + r1, y:y + c1] = dst
    print('res')
    cv2.imshow('res', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bg

def bg_merge_fg(bg, fg, x, y):
    r1, c1, _ = fg.shape
    roi = bg[x:x + r1, y:y + c1]
    # dst = np.zeros((r1, c1, 3), np.uint8)
    for i in range(r1):
        for j in range(c1):
            (b, g, r) = fg[i, j]
            if (b, g, r) == (255, 255, 255):  # template（fg）白色背景n
                # ：
                fg[i, j] = roi[i, j]
    dst = cv2.addWeighted(roi, 0.5, fg, 0.5, 0)
    bg[x:x + r1, y:y + c1] = dst
    print('dst')
    cv2.imshow('dst', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bg

def face_mask(src, face, m, n):
    r0, c0 = src.shape  # 320, 252
    r1, c1 = face.shape  # 196, 181
    roi = src[m:m + r1, n:n + c1]
    ret, fmask = cv2.threshold(face, 0, 255, cv2.THRESH_BINARY)
    # ROI掩模区域反向掩模
    fmask_inv = cv2.bitwise_not(fmask)

    # 掩模显示前景
    # Take only region of logo from logo image.
    src_fg = cv2.bitwise_and(roi, roi, mask=fmask_inv)
    print('src_fg')
    cv2.imshow('src_fg', src_fg)
    cv2.waitKey(0)

    src[m:m + r1, n:n + c1] = src_fg
    print('img_mask')
    cv2.imshow('ima_mask', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return src

def imgconvert(imag_path, save_path, flag='L'):
    imgsrc = Image.open(imag_path)
    print(imgsrc.getbands())
    imgret = imgsrc.convert(flag)
    print(imgret.getbands())
    imgret.save(save_path)

def bgaddedge(imag_path, save_path):
    imgsrc = cv2.imread(imag_path, 1)
    height, width, _ = imgsrc.shape
    dst = np.zeros((height, width, 3), np.uint8)
    for h in range(0, height):
        for j in range(0, width):
            if h <= 10 or h >= height-11 or j <= 10 or j >= width-11:
                imgsrc[h, j] = (255, 255, 255)  # 白色
            dst[h, j] = imgsrc[h, j]
    cv2.imwrite(save_path, dst)

def create_datatree():
    template_path = './samples/template'
    image_path = './samples/image'
    fake_path = './samples/Samples_right/'
    true_path = './samples/Samples_left/'
    names = os.listdir(fake_path)
    for name in names:
        sub_tpath = os.path.join(template_path, name[:-4])
        sub_ipath = os.path.join(image_path, name[:-4])
        if not os.path.exists(sub_tpath):
            os.makedirs(sub_tpath)
        if not os.path.exists(sub_ipath):
            os.makedirs(sub_ipath)
        shutil.copy(true_path+name, sub_tpath)
        new_file = os.path.join(sub_ipath, name[:-4]+'_fake.png')
        shutil.copy(fake_path+name, new_file)
        shutil.copy('./samples/Samples_left/'+name, sub_ipath)


if __name__ == '__main__':
    needface = False
    imag_path = './example/template/T5/000016_cut2.png'
    save_path = './example/template/T5/000016_cut.png'
    sample_path = './QATM/sample/sample1.jpg'
    create_datatree()
    sys.exit()
    # bgaddedge(save_path, save_path)
    # img1 = Image.open(imag_path)
    # img2 = Image.open(save_path)
    # 前两个坐标点是左上角坐标
    # 后两个坐标点是右下角坐标
    # width在前， height在后
    box = (60, 100, 300, 250)
    # region = img1.crop(box)
    # region.save(save_path)
    bgaddedge(save_path, imag_path)
    # imgconvert(save_path, save_path, 'P')
    # imgdst = cv2.resize(imgsrc, (2*imgsrc.shape[1], 2*imgsrc.shape[0]))
    # cv2.imwrite(save_path, imgdst)
    sys.exit()
    # 图像输入
    bg = cv2.imread('./example/image/I4/face_24.png', 1)
    src = cv2.imread('./example/template/T3_4/image1c.png', 1)
    face = cv2.imread("./example/face_gray.png", 0)
    r0, c0, _ = bg.shape
    r1, c1, _ = src.shape
    r2, c2 = face.shape

    # I want to put logo on top-left corner, So I create a ROI
    # 首先获取原始图像roi
    x = np.random.randint(0, r0-r1)
    y = np.random.randint(0, c0-c1)

    # img = bg_add_fg(bg, src, x, y)
    img = bg_merge_fg(bg, src, x, y)
    if needface:  # face遮挡
        m = np.random.randint(0, r0-r2)
        n = np.random.randint(0, c0-c2)
        img = face_mask(img, face, m, n)
    # sys.exit()
    getstr = input("[y/n]")
    if getstr[0] == 'y':
        cv2.imwrite('./example/image/I4/image'+getstr[1:]+'.png', img)
