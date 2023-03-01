from qatm import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化

def plt_show(id_list, y1, y2):
    plt.figure(1)
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
    colums = cache.keys()
    data = cache.values()
    save = pd.DataFrame(columns=colums, data=data)
    f1 = open(path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk')
    f1.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()

    template_path = '../samples/template'
    image_path = '../samples/image'
    names = os.listdir(template_path)
    cache = {
        'true_score': [],
        'fake_score': [],
        'true-fake': []
    }
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    for name in names:
        template_dir = os.path.join(template_path, name)
        sub_path1 = os.path.join(name, name + '.png')
        sub_path2 = os.path.join(name, name + '_fake.png')
        image_path1 = os.path.join(image_path, sub_path1)
        image_path2 = os.path.join(image_path, sub_path2)
        dataset1 = ImageDataset(Path(template_dir), image_path1, thresh_csv=args.thresh_csv)
        dataset2 = ImageDataset(Path(template_dir), image_path2, thresh_csv=args.thresh_csv)
        # print("define model...")
        # print("calculate score...")
        scores1, w_array1, h_array1, thresh_list1 = run_multi_sample(model, dataset1)
        # print(scores)
        # print("nms...")
        _, _, max_score1 = nms_multi(scores1, w_array1, h_array1, thresh_list1)
        cache['true_score'].append(np.array(max_score1))
        scores2, w_array2, h_array2, thresh_list2 = run_multi_sample(model, dataset2)
        # print(scores)
        # print("nms...")
        _, _, max_score2 = nms_multi(scores2, w_array2, h_array2, thresh_list2)
        cache['fake_score'].append(np.array(max_score2))
        cache['true-fake'].append(np.array(max_score1 - max_score2))
        if len(cache['fake_score']) >= 3:
            break

    id_list = range(0, len(cache['true_score']))
    save_data(cache, '../output/all.csv')
    plt_show(id_list, cache['true_score'], cache['fake_score'])
    avg = np.array(cache['true-fake']).absolute().mean()
    print('mean_delta:{}'.format(avg))
