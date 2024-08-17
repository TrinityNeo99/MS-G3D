#  Copyright (c) 2023-2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'kinetics', 'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset',
                                 'pingpong-109-coco', "ntu-60-256-xview", "ntu-60-256-xsub"},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')

    arg = parser.parse_args()

    dataset = arg.dataset
    try:
        val_label_path = './data/' + dataset + '/val_label.pkl'
        if dataset == "ntu-60-256-xview":
            val_label_path = "../dataset/ntu-60-256/xview/val_label.pkl"
        elif dataset == "ntu-60-256-xsub":
            val_label_path = "../dataset/ntu-60-256/xsub/val_label.pkl"
        with open(val_label_path, 'rb') as label:
            label = np.array(pickle.load(label))
    except FileNotFoundError:  # 当没找到ntu数据时，../dataset 找到数据
        with open('../dataset/2023-3-29_北体合作_示范动作/MS-G3D/' + dataset + '/val_label.pkl', 'rb') as label:
            label = np.array(pickle.load(label))

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        r = r11 + r22 * arg.alpha
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
