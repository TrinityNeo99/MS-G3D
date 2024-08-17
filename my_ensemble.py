#  Copyright (c) 2023-2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-label-path',
                        required=True, )
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)
    parser.add_argument('--joint-val-result-path',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-val-result-path',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')

    arg = parser.parse_args()

    with open(arg.val_label_path, 'rb') as label:
        label = np.array(pickle.load(label))

    with open(arg.joint_val_result_path, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.bone_val_result_path, 'rb') as r2:
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
