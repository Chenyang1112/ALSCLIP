from __future__ import print_function
import random
import shutil
import os
import glob
from copy import deepcopy

import torch  # why is it located here?
import numpy as np
from plyfile import PlyData
import pdb
import cv2

cv2.setNumThreads(0)
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import yaml
from easydict import EasyDict
from dataset.loader_s3dis import test
from utils.my_args import my_args
from utils.common_util import AverageMeter, intersectionAndUnion, find_free_port
from model import get as get_model
from dataset import get as get_dataset


seed = 0
pl.seed_everything(seed)  # , workers=True
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


# https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_mnist.py
def cli_main():
    # ------------
    # args
    # ------------
    parser = my_args()
    args = parser.parse_args()

    # ------------
    # randomness or seed
    # ------------
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # ------------
    # logger
    # ------------
    # from pytorch_lightning.loggers import NeptuneLogger
    # neptune_path = os.path.join(args.MYCHECKPOINT, 'neptune.npz')
    # if args.neptune_id:
    #     np.savez(
    #         neptune_path,
    #         project=args.neptune_proj,
    #         id=args.neptune_id)
    #     print(" >> newly create naptune.npz in test_pl.py")
    #
    # if os.path.exists(neptune_path):
    #     neptune_info = np.load(neptune_path)
    #
    #     neptune_logger = NeptuneLogger(
    #         api_key="<YOUR NEPTUNE>",
    #         project=args.neptune_proj)
    #     neptune_logger._run_short_id = str(neptune_info['id'])
    #     print(">> re-use the neptune: id[%s]"%(neptune_info['id']))
    # else:
    #     neptune_logger = NeptuneLogger(
    #         api_key="<YOUR NEPTUNE>",
    #         project=args.neptune_proj)
    #     print(">> start new neptune")
    #
    # neptune_logger.experiment["sys/tags"].add('test_pl.py')

    # ------------
    # model
    # ------------
    ckpts = sorted(glob.glob(os.path.join(args.MYCHECKPOINT, "*.ckpt")))
    if len(ckpts) > 1: ckpts = ckpts[:-1]  # remove 'last.ckpt'
    mIoU_val_best = -1.
    filename_best = None
    for ckpt in ckpts:
        rootpath = '/'.join(ckpt.split('/')[:-1])
        filename = ckpt.split('/')[-1]
        mIoU_val = filename[:-5]
        mIoU_val = float((mIoU_val.split('--')[-2])[9:])
        if mIoU_val >= mIoU_val_best:
            mIoU_val_best = mIoU_val
            filename_best = filename
    ckpt_best = os.path.join(rootpath, filename_best)
    args.load_model = ckpt_best
    args.on_train = False
    print('ckpt best. args.load_model=[{}]'.format(args.load_model))
    assert args.load_model is not None, 'why did you come?'
    model = get_model(args.model).load_from_checkpoint(
        os.path.join(args.load_model),
        args=args,
        strict=True)  # args.strict_load

    model.eval()
    model.freeze()

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(

        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),  # 'ddp'
        enable_progress_bar=False if 'NVIDIA' in args.computer else True,
    )

    # ------------
    # test
    # ------------
    if args.dataset == 'loader_s3dis':
        data_root = os.path.join(args.s3dis_root, 'trainval_fullarea')
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]


    ckpt_name = (args.load_model).split('/')[-1]
    ckpt_name = ckpt_name[:-5]  # remove '.ckpt'
    save_folder = os.path.join(args.MYCHECKPOINT, 'test_results__%s' % ((ckpt_name)))
    os.makedirs(save_folder, exist_ok=True)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_save, label_save = [], []

    str_to_log = '<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<'
    print(str_to_log)

    for idx, item in enumerate(data_list):

        pred_save_filename = \
            '{}__epoch_{}npts{:09d}__size0p{:04d}__pred__test_pl__.npy'.format(
                item, model.current_epoch, args.eval_voxel_max, int(args.voxel_size * 10000))
        pred_save_path = os.path.join(save_folder, pred_save_filename)

        label_save_filename = \
            '{}__epoch_{}npts{:09d}__size0p{:04d}__label__test_pl__.npy'.format(
                item, model.current_epoch, args.eval_voxel_max, int(args.voxel_size * 10000))
        label_save_path = os.path.join(save_folder, label_save_filename)

        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            print('{}/{}: [{}], loaded pred and label.'.format(
                idx + 1, len(data_list), item))
            pred = np.load(pred_save_path)
            label = np.load(label_save_path)

        else:

            if args.dataset == 'loader_s3dis':
                data_path = os.path.join(
                    args.s3dis_root, 'trainval_fullarea', item + '.npy')
                data = np.load(data_path)
                label = data[:, 5]  # coord, feat = data[:, :3], data[:, 3:6]
                mode_eval = 'test'



            with torch.no_grad():
                model.pred = torch.zeros((label.size, args.classes)).cuda()

            dataset = get_dataset(args.dataset)
            test_loader_kwargs = \
                {
                    "batch_size": args.test_batch,  # WRONG. Because of my stupid code. ,
                    "num_workers": args.val_worker,
                    "collate_fn": dataset.TestCollateFn,
                    "pin_memory": False,
                    "drop_last": False,
                    "shuffle": False,
                }

            test_loader = torch.utils.data.DataLoader(
                test(args, mode=mode_eval, test_split=item),
                **test_loader_kwargs)

            trainer.test(model=model, dataloaders=test_loader, verbose=True)

            pred = model.pred.max(1)[1].cpu().detach().numpy()


            np.save(pred_save_path, np.hstack((data[:,:3],pred.reshape(-1,1))))
            np.save(label_save_path, np.hstack((data[:,:3],label.reshape(-1,1))))
            # end of it cond

        # calculation 1: add per room predictions
        intersection, union, target = \
            intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        # 计算 F1 指数

        f1_class = 2 * intersection / (intersection + union + 1e-10)

        f1_mean = np.mean(f1_class)

        str_to_log = \
            'Test: [{:4d}/{:4d}]-npts[{npts:7d}/{:7d}] Accuracy[{accuracy:.4f}  F1score[{f1_mean:.4f}]'.format(
                int(idx + 1), len(data_list), int(label.size),
                accuracy=accuracy,
                f1_mean=f1_mean,
                npts=args.eval_voxel_max)
        print(str_to_log)

        pred_save.append(pred)
        label_save.append(label)
        # end of the for loop. (per-scene)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)



    # calculation 2
    intersection, union, target = \
        intersectionAndUnion(
            np.concatenate(pred_save),
            np.concatenate(label_save),
            args.classes,
            args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)

    # 计算 F1 指数

    f1_class = 2 * intersection / (intersection + union + 1e-10)

    f1_mean = np.mean(f1_class)

    # 打印结果
    str_to_log = \
        'Val result: mIoU/mAcc/allAcc/F1 {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
            mIoU, mAcc, allAcc, f1_mean)
    print(str_to_log)




    if args.dataset == 'loader_s3dis':
        names_path = os.path.join(args.s3dis_root, "list/s3dis_names.txt")
        names = [line.rstrip('\n') for line in open(names_path)]
    elif args.dataset == 'loader_scannet':
        names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    else:
        print("Please set labels' names.")
        raise NotImplemented

    for i in range(args.classes):

        str_to_log = \
            'Class_{} Result: iou/accuracy/f1_mean {:.4f}/{:.4f}/{:.4f}, name: {}.'.format(
                i, iou_class[i], accuracy_class[i], f1_class[i],names[i])
        print(str_to_log)


    str_to_log = '<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<'
    print(str_to_log)





if __name__ == '__main__':
    cli_main()


