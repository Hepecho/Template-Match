from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *

# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys

with open("qatm_pytorch.py", encoding='utf-8') as f:
       p = ast.parse(f.read())

for node in p.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        p.body.remove(node)

module = types.ModuleType("mod")
code = compile(p, "mod.py", 'exec')
sys.modules["mod"] = module
exec(code,  module.__dict__)

from mod import *
# -

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image', default='../samples_multi/real_mask/000000.png')
    parser.add_argument('-t', '--template_images_dir', default='../samples_multi/templates/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    
    template_dir = args.template_images_dir
    image_path = args.sample_image
    dataset = ImageDataset(Path(template_dir), image_path, thresh_csv=args.thresh_csv)
    
    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=True)
    print("calculate score...")
    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)  # w_array和h_array表示template的尺寸
    # score [t, s_h, s_w]   w_array h_array thresh_list [t,]
    # print(scores)
    print("nms...")
    boxes, indices, max_score, mean_score, shape = nms_multi(scores, w_array, h_array, thresh_list)
    print("multi:")
    print(max_score)
    print(mean_score)
    print(shape)
    _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='../output/result_6.png')
    print("result.png was saved")


