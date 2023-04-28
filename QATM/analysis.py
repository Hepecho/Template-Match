from qatm import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import csv
import scipy.stats
import seaborn as sns
import time
from sklearn.metrics import precision_recall_curve
import numpy as np


plt.rcParams['axes.unicode_minus'] = False
colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
np.seterr(divide='ignore',invalid='ignore')

def plt_show(id_list, y1, y2):
    plt.figure(figsize=(8, 4))
    color = mcolors.TABLEAU_COLORS[colors[1]]
    # 点线图
    plt.plot(id_list, y1, linestyle='-', color=color, label='true')
    color = mcolors.TABLEAU_COLORS[colors[2]]
    plt.plot(id_list, y2, linestyle='-', color=color, label='fake')
    # 点图
    plt.legend()
    plt.xlabel('id')
    plt.ylabel('max_score')
    # x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # plt.gca().xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    # plt.xlim(0.5, 20.5)
    plt.show()

def save_data(cache, path):
    colums = list(cache.keys())
    values = list(cache.values())
    values_T = list(map(list, zip(*values)))
    save = pd.DataFrame(columns=colums, data=values_T)
    f1 = open(path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk')
    f1.close()

def read_data(cache, path):
    with open(path) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        next(rows)  # 读取首行
        # 读取除首行之后每一行的数据
        for r in rows:
            cache['real_score'].append(float(r[1]))  # 将字符串数据转化为浮点型加入到数组之中
            cache['fake_score'].append(float(r[2]))
            cache['real-fake'].append(float(r[3]))


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)


def best_threshold(label, predict):
    precisions, recalls, thresholds = precision_recall_curve(label, predict)

    # 拿到最优结果以及索引
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    # 阈值
    print("best_threshold:", thresholds[best_f1_score_index])
    print("best_f1_score:", best_f1_score)
    print("precision:", precisions[best_f1_score_index])
    print("recall:", recalls[best_f1_score_index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default=None)
    parser.add_argument('-cal', '--calculate', action='store_true')
    parser.add_argument('-o', '--output', default='../output/all.csv')
    parser.add_argument('-w', '--way', default='qatm')
    parser.add_argument('-p', '--pix', type=int, default=2)
    args = parser.parse_args()

    parent_path = '../samples_multi10_p6'
    template_path = os.path.join(parent_path, 'templates_2pix_b')
    real_path = os.path.join(parent_path, 'real_mask')
    fake_path = os.path.join(parent_path, 'fake_mask')
    names = os.listdir(template_path)
    cache = {
        'real_score': [],
        'fake_score': [],
        'real-fake': []
    }
    pix = args.pix - 2

    if args.calculate:
        start = time.process_time()
        if args.way == 'qatm':
            model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
            for name in names:
                # print(name)
                template_dir = os.path.join(template_path, name)
                real_image = os.path.join(real_path, name+'.png')
                fake_image = os.path.join(fake_path, name+'.png')
                real_dataset = ImageDataset(Path(template_dir), real_image, thresh_csv=args.thresh_csv)
                fake_dataset = ImageDataset(Path(template_dir), fake_image, thresh_csv=args.thresh_csv)
                # print("define model...")
                # print("calculate score...")
                real_score, real_w_array, real_h_array, real_thresh_list = run_multi_sample(model, real_dataset)
                # print(scores)
                # print("nms...")
                _, _, real_max_score, real_mean_score, _ = nms_multi(real_score, real_w_array, real_h_array, real_thresh_list)
                cache['real_score'].append(np.array(real_max_score))
                fake_scores, fake_w_array, fake_h_array, fake_thresh_list = run_multi_sample(model, fake_dataset)
                # print(scores)
                # print("nms...")
                _, _, fake_max_score, fake_mean_score, _ = nms_multi(fake_scores, fake_w_array, fake_h_array, fake_thresh_list)
                cache['fake_score'].append(np.array(fake_max_score))
                cache['real-fake'].append(np.array(real_max_score - fake_max_score))
                # if len(cache['fake_score']) >= 10:
                # break
        else:
            for name in names:
                # print(name)
                template_path_ = os.path.join(template_path, name, name + '.png')
                template = cv2.imread(template_path_)
                template = cv2.copyMakeBorder(template, pix, pix, pix, pix, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                real_image_path = os.path.join(real_path, name + '.png')
                real_image = cv2.imread(real_image_path)
                fake_image_path = os.path.join(fake_path, name + '.png')
                fake_image = cv2.imread(fake_image_path)

                real_res = cv2.matchTemplate(real_image, template, cv2.TM_CCOEFF_NORMED)
                _, real_max_score, _, _ = cv2.minMaxLoc(real_res)
                cache['real_score'].append(np.array(real_max_score))
                fake_res = cv2.matchTemplate(fake_image, template, cv2.TM_CCOEFF_NORMED)
                _, fake_max_score, _, _ = cv2.minMaxLoc(fake_res)
                cache['fake_score'].append(np.array(fake_max_score))
                cache['real-fake'].append(np.array(real_max_score - fake_max_score))

        end = time.process_time()
        print('running time:%sms' % ((end - start) * 1000))
        save_data(cache, args.output)

    else:
        read_data(cache, args.output)
        plt_show(range(0, len(cache['real_score'])), cache['real_score'], cache['fake_score'])

        predict = []  # 用于计算最佳阈值
        label = []
        for k in cache.keys():
            print(k+':')
            print('var:{}'.format(np.var(cache[k])))  # 输出方差
            print('mean:{}'.format(np.mean(cache[k])))  # 输出均值
            if k == 'real_score':
                label.extend(np.ones(len(cache[k])))
                predict.extend(cache[k])
            elif k == 'fake_score':
                label.extend(np.zeros(len(cache[k])))
                predict.extend(cache[k])

        best_threshold(label, predict)  # 计算最佳阈值
        print("Js散度：")  # JS散度基于KL散度，同样是二者越相似，JS散度越小
        print(JS_divergence(np.array(cache['real_score']), np.array(cache['fake_score'])))  # 0.004463665396105692

        # 样本数据密度分布图
        plt.figure(figsize=(8, 4))
        # plt.xlim((-0.01, 0.07))
        # plt.ylim((0, 220))
        true_color = mcolors.TABLEAU_COLORS[colors[1]]
        fake_color = mcolors.TABLEAU_COLORS[colors[2]]
        sns.kdeplot(cache['fake_score'], label="fake_score", linestyle='--', color=fake_color).legend(loc="upper right")
        sns.kdeplot(cache['real_score'], label="real_score", color=true_color).legend(loc="upper right")
        # bw → 也可以类似看做直方图的箱数，数越大，箱子越多，刻画的越精确。
        plt.show()


