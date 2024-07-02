#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: angular_feature.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/5/5 21:09 at PyCharm
"""
import torch
import torch.nn as nn
from einops import rearrange

pingpong_coco_bone_angle_pairs = {
    3: (1, 0),
    1: (0, 3),
    4: (2, 0),
    2: (0, 4),
    0: (5, 6),
    6: (8, 5),
    5: (7, 6),
    7: (9, 5),
    8: (6, 10),
    10: (10, 10),
    11: (5, 13),
    12: (6, 14),
    13: (11, 15),
    15: (15, 15),
    14: (12, 16),
    16: (16, 16),
    9: (9, 9)
}

pingpong_coco_bone_adj = {
    3: 1,
    1: 0,
    4: 2,
    2: 0,
    0: 5,
    6: 0,
    5: 7,
    7: 9,
    8: 6,
    10: 8,
    11: 5,
    12: 6,
    13: 11,
    15: 13,
    14: 12,
    16: 14,
    9: 7,  # add
    # 5: 6,
    # 11: 12,
}

ntu_bone_angle_pairs = {
    25: (24, 12),
    24: (25, 12),
    12: (24, 25),
    11: (12, 10),
    10: (11, 9),
    9: (10, 21),
    21: (9, 5),
    5: (21, 6),
    6: (5, 7),
    7: (6, 8),
    8: (23, 22),
    22: (8, 23),
    23: (8, 22),
    3: (4, 21),
    4: (4, 4),
    2: (21, 1),
    1: (17, 13),
    17: (18, 1),
    18: (19, 17),
    19: (20, 18),
    20: (20, 20),
    13: (1, 14),
    14: (13, 15),
    15: (14, 16),
    16: (16, 16)
}

ntu_bone_adj = {
    25: 12,
    24: 12,
    12: 11,
    11: 10,
    10: 9,
    9: 21,
    21: 21,
    5: 21,
    6: 5,
    7: 6,
    8: 7,
    22: 8,
    23: 8,
    3: 21,
    4: 3,
    2: 21,
    1: 2,
    17: 1,
    18: 17,
    19: 18,
    20: 19,
    13: 1,
    14: 13,
    15: 14,
    16: 15
}


class Angular_feature:
    def __init__(self):
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

    def preprocessing_pingpong_coco(self, x, addMotion=False):
        # Extract Bone and Angular Features
        fp_sp_joint_list_bone = []
        fp_sp_joint_list_bone_angle = []
        fp_sp_two_hand_angle = []
        fp_sp_two_elbow_angle = []
        fp_sp_two_knee_angle = []
        fp_sp_two_feet_angle = []

        all_list = [
            fp_sp_joint_list_bone, fp_sp_joint_list_bone_angle,
            fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_knee_angle,
            fp_sp_two_feet_angle, fp_sp_two_hand_angle,
        ]  # 3 + 1 + 1 + 1 + 1 + 1
        # print("len of pairs.keys", len(pingpong_coco_bone_angle_pairs.keys()))
        for a_key in pingpong_coco_bone_angle_pairs.keys():
            a_angle_value = pingpong_coco_bone_angle_pairs[a_key]
            a_bone_value = pingpong_coco_bone_adj[a_key]
            the_joint = a_key
            a_adj = a_bone_value
            bone_diff = (x[:, :3, :, the_joint, :] -
                         x[:, :3, :, a_adj, :]).unsqueeze(3)
            fp_sp_joint_list_bone.append(bone_diff)

            # bone angles
            v1 = a_angle_value[0]
            v2 = a_angle_value[1]
            vec1 = x[:, :3, :, v1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, v2, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two hand angle
            vec1 = x[:, :3, :, 9, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 10, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two elbow angle
            vec1 = x[:, :3, :, 7, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 8, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two knee angle
            vec1 = x[:, :3, :, 11, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 12, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_knee_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two feet angle
            vec1 = x[:, :3, :, 13, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 14, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_feet_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

        for a_list_id in range(len(all_list)):
            all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)  # cat along V dimension

        # generate motion feature
        if addMotion:
            motion = []
            N, C, T, V, M = x.size()
            motion_diff = x[:, :3, 0, :, :] - x[:, :3, 0, :, :]
            for i in range(1, T):
                motion_diff = x[:, :3, i, :, :] - x[:, :3, i - 1, :, :]
                motion.append(motion_diff.unsqueeze(2))
            motion.append(motion_diff.unsqueeze(2))  # the last one equal to the before
            motion = torch.cat(motion, dim=2)  # cat along T dimension
            all_list.append(motion)  # add motion feature
        all_list = torch.cat(all_list, dim=1)  # cat along feature C dimension
        features = torch.cat((x, all_list.cuda(x.device)), dim=1)
        return features

    def preprocessing_pingpong_coco_upper_body(self, x):
        # Extract Bone and Angular Features
        fp_sp_joint_list_bone = []
        fp_sp_joint_list_bone_angle = []
        fp_sp_two_hand_angle = []
        fp_sp_two_elbow_angle = []

        all_list = [
            fp_sp_joint_list_bone, fp_sp_joint_list_bone_angle,
            fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_hand_angle
        ]  # 3 + 1 + 1 + 1 + 1 + 1
        # print("len of pairs.keys", len(pingpong_coco_bone_angle_pairs.keys()))
        for a_key in pingpong_coco_bone_angle_pairs.keys():
            a_angle_value = pingpong_coco_bone_angle_pairs[a_key]
            a_bone_value = pingpong_coco_bone_adj[a_key]
            the_joint = a_key
            a_adj = a_bone_value
            bone_diff = (x[:, :3, :, the_joint, :] -
                         x[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
            fp_sp_joint_list_bone.append(bone_diff)

            # bone angles
            v1 = a_angle_value[0]
            v2 = a_angle_value[1]
            vec1 = x[:, :3, :, v1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, v2, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # two hand angle
            vec1 = x[:, :3, :, 9, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 10, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # two elbow angle
            vec1 = x[:, :3, :, 7, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 8, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

        for a_list_id in range(len(all_list)):
            all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)
            # print(all_list[a_list_id].shape)

        # print("a_key: ", a_key)
        all_list = torch.cat(all_list, dim=1)
        # print('All_list:', all_list.shape)
        #
        # print('x:', x.shape)

        features = torch.cat((x, all_list.cuda()), dim=1)
        # print('features:', features.shape)
        return features

    def ntu_preprocessing(self, x):
        # Extract Bone and Angular Features
        fp_sp_joint_list_bone = []
        fp_sp_joint_list_bone_angle = []
        fp_sp_joint_list_body_center_angle_1 = []
        fp_sp_joint_list_body_center_angle_2 = []
        fp_sp_left_hand_angle = []
        fp_sp_right_hand_angle = []
        fp_sp_two_hand_angle = []
        fp_sp_two_elbow_angle = []
        fp_sp_two_knee_angle = []
        fp_sp_two_feet_angle = []

        all_list = [
            fp_sp_joint_list_bone, fp_sp_joint_list_bone_angle, fp_sp_joint_list_body_center_angle_1,
            fp_sp_joint_list_body_center_angle_2, fp_sp_left_hand_angle, fp_sp_right_hand_angle,
            fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_knee_angle,
            fp_sp_two_feet_angle
        ]

        for a_key in ntu_bone_angle_pairs:
            a_angle_value = ntu_bone_angle_pairs[a_key]
            a_bone_value = ntu_bone_adj[a_key]
            the_joint = a_key - 1
            a_adj = a_bone_value - 1
            bone_diff = (x[:, :3, :, the_joint, :] -
                         x[:, :3, :, a_adj, :]).unsqueeze(3)
            fp_sp_joint_list_bone.append(bone_diff)

            # bone angles
            v1 = a_angle_value[0] - 1
            v2 = a_angle_value[1] - 1
            vec1 = x[:, :3, :, v1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, v2, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # body angles 1
            vec1 = x[:, :3, :, 2 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 21 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_body_center_angle_1.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # body angles 2
            vec1 = x[:, :3, :, the_joint, :] - x[:, :3, :, 21 - 1, :]
            vec2 = x[:, :3, :, 2 - 1, :] - x[:, :3, :, 21 - 1, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_body_center_angle_2.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # left hand angle
            vec1 = x[:, :3, :, 24 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 25 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_left_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # right hand angle
            vec1 = x[:, :3, :, 22 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 23 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_right_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two hand angle
            vec1 = x[:, :3, :, 24 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 22 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two elbow angle
            vec1 = x[:, :3, :, 10 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 6 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two knee angle
            vec1 = x[:, :3, :, 18 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 14 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_knee_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

            # two feet angle
            vec1 = x[:, :3, :, 20 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 16 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_feet_angle.append(angular_feature.unsqueeze(2).unsqueeze(1))

        for a_list_id in range(len(all_list)):
            all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)

        all_list = torch.cat(all_list, dim=1)
        features = torch.cat((x, all_list.cuda(x.device)), dim=1)
        return features
