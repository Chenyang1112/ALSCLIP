import os
import random
import pdb
import glob

import numpy as np
from plyfile import PlyData

import torch
import torch.utils.data

from dataset.utils.voxelize import voxelize
from utils.my_args import my_args

seed = 0
# pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU


# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def get_parser():
    parser = my_args()
    args = parser.parse_args()
    return args


def main():
    global args
    args = get_parser()
    test()


def data_load(data_name):
    if args.dataset == 'loader_s3dis':
        data_path = os.path.join(
            args.s3dis_root,
            'trainval_fullarea',
            data_name + '.npy')
        data = np.load(data_path)  # xyzrgbl, N*7
        # coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

        points = data[:, :5]
        labels = data[:, 5]
        block_size = 40
        stride = 20
        padding = 2
        block_points = 4096
        idx_data = []
        coord, feat, label, = [], [], []
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * stride
                e_x = min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size
                s_y = coord_min[1] + index_y * stride
                e_y = min(s_y + block_size, coord_max[1])
                s_y = e_y - block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - padding) & (points[:, 0] <= e_x + padding) & (
                                points[:, 1] >= s_y - padding) & (
                            points[:, 1] <= e_y + padding))[0]
                if point_idxs.size == 0:
                    continue

                num_batch = int(np.ceil(point_idxs.size / block_points))
                point_size = int(num_batch * block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))

                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
                # data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                # idx_data.append(point_idxs)
                # label.append(label_batch)
                # coord.append(data_batch[:, :3])
                # feat.append(data_batch[:, 3:6])

    #             data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
    #             label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
    #
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
    #     data_room = data_room.reshape((-1, block_points, data_room.shape[1]))
    #     label_room = label_room.reshape((-1, block_points))
    #     sample_weight = sample_weight.reshape((-1, block_points))
        index_room = index_room.reshape((-1, block_points))
        idx_data=[]

        for i in range(index_room.shape[0]):
            idx_data.append(index_room[i,:])


    #     return data_room, label_room,  index_room
    # idx_data = []
    # idx_data.append(np.arange(label.shape[0]))
    coord, feat, label = data[:, :3], data[:, 3:5], data[:, 5]
    return coord, feat, label, idx_data


def test():
    if args.dataset == 'loader_s3dis':
        foldpath = os.path.join(args.s3dis_root, 'test_split')
        os.makedirs(foldpath, exist_ok=True)
        # data_list = data_prepare()
        data_root = os.path.join(args.s3dis_root, 'trainval_fullarea')
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if
                     'Area_{}'.format(args.test_area) in item]
    print("Totally {} samples in val set.".format(len(data_list)))

    for idx, item in enumerate(data_list):

        coord, feat, label, idx_data = data_load(item)
        idx_size = len(idx_data)
        idx_list, coord_list, feat_list, offset_list, label_list = \
            [], [], [], [], []
        for i in range(idx_size):
            idx_part = idx_data[i]
            coord_part = coord[idx_part]
            feat_part = feat[idx_part]
            label_part = label[idx_part]
            # coord_part, feat_part = input_normalize(coord_part, feat_part)

            idx_list.append(idx_part)
            coord_list.append(coord_part)
            feat_list.append(feat_part)
            offset_list.append(idx_part.size)
            label_list.append(label_part)

            # end of the for loop. (single batch per scene)

        for idx_batch in range(len(idx_list)):

            idx_part = idx_list[idx_batch]
            coord_part = coord_list[idx_batch]
            feat_part = feat_list[idx_batch]
            offset_part = offset_list[idx_batch]
            label_part = label_list[idx_batch]

            if args.dataset == 'loader_s3dis':
                filename = '{}__idx_{:05d}__npts_{:09d}.npz'.format(
                    item,
                    idx_batch,
                    args.eval_voxel_max)
            elif args.dataset == 'loader_scannet':
                filename = '{}__idx_{:05d}__npts_{:09d}__size0p{:04d}.npz'.format(
                    item,
                    idx_batch,
                    args.eval_voxel_max,
                    int(args.voxel_size * 10000))
            else:
                raise NotImplemented
            filepath = os.path.join(foldpath, filename)

            print("Save files [%d/%d:%s] [%d/%d] coord_part.shape[0]=[%d] voxel_size[%.4f], eval_voxel_max[%d]" % (
                idx + 1, len(data_list), filepath,
                idx_batch, len(idx_list),
                coord_part.shape[0],
                args.voxel_size,
                args.eval_voxel_max))

            np.savez(
                filepath,
                idx_part=idx_part,
                coord_part=coord_part,
                feat_part=feat_part,
                offset_part=offset_part,
                label_part=label_part)


if __name__ == '__main__':
    main()

