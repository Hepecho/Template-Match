from qatm import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import csv
import scipy.stats
import seaborn as sns
import time

plt.rcParams['axes.unicode_minus'] = False
colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化

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
            if float(r[1]) < 0.00006:
                print(float(r[1]))
            cache['fake_score'].append(float(r[2]))
            cache['real-fake'].append(float(r[3]))

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default=None)
    parser.add_argument('-cal', '--calculate', action='store_true')
    parser.add_argument('-o', '--output', default='../output/all.csv')
    args = parser.parse_args()

    parent_path = '../samples_multi'
    template_path = os.path.join(parent_path, 'templates_PPP')
    real_path = os.path.join(parent_path, 'real_mask')
    fake_path = os.path.join(parent_path, 'fake_mask')
    names = os.listdir(template_path)
    cache = {
        'real_score': [],
        'fake_score': [],
        'real-fake': []
    }
    if args.calculate:
        start = time.process_time()
        model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
        for name in names:
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

        end = time.process_time()
        print('running time:%sms' % ((end - start) * 1000))
        save_data(cache, args.output)

    else:
        read_data(cache, args.output)
        plt_show(range(0, len(cache['real_score'])), cache['real_score'], cache['fake_score'])
        for k in cache.keys():
            print(k+':')
            print(np.var(cache[k]))  # 输出方差
            print(np.mean(cache[k]))  # 输出均值
        print("Js散度：")  # JS散度基于KL散度，同样是二者越相似，JS散度越小
        print(JS_divergence(np.array(cache['real_score']), np.array(cache['fake_score'])))  # 0.004463665396105692

        # 样本数据密度分布图
        plt.figure(figsize=(8, 4))
        sns.kdeplot(cache['fake_score'], label="fake_score", linestyle='--').legend(loc="upper right")
        sns.kdeplot(cache['real_score'], label="real_score").legend(loc="upper right")
        # bw → 也可以类似看做直方图的箱数，数越大，箱子越多，刻画的越精确。
        plt.show()


