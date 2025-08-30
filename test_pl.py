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



# from __future__ import print_function
# import random
# import shutil
# import os
# import glob
# from copy import deepcopy
#
# import torch # why is it located here?
# import numpy as np
# from plyfile import PlyData
# import pdb
# import cv2
# cv2.setNumThreads(0)
# import pytorch_lightning as pl
# from pytorch_lightning.strategies import DDPStrategy
# import yaml
# from easydict import EasyDict
# from dataset.loader_s3dis import ScannetDatasetWholeScene
# from utils.my_args import my_args
# from utils.common_util import AverageMeter, intersectionAndUnion, find_free_port
# from model import get as get_model
# from dataset import get as get_dataset
# from tqdm import tqdm
# seed=0
# pl.seed_everything(seed) # , workers=True
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if use multi-GPU
# # torch.backends.cudnn.deterministic=True
# # torch.backends.cudnn.benchmark=False
#
# def read_txt(path):
#     with open(path) as f:
#         lines = f.readlines()
#     lines = [x.strip() for x in lines]
#     return lines
#
# def add_vote(vote_label_pool, point_idx, pred_label, weight):
#     B = pred_label.shape[0]
#     N = pred_label.shape[1]
#     for b in range(B):
#         for n in range(N):
#             if weight[b,n]:
#                 vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
#     return vote_label_pool
#
# # https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_mnist.py
# def cli_main():
#
#     # ------------
#     # args
#     # ------------
#     parser = my_args()
#     args = parser.parse_args()
#
#     # ------------
#     # randomness or seed
#     # ------------
#     torch.backends.cudnn.benchmark = args.cudnn_benchmark
#
#     # ------------
#     # logger
#     # ------------
#     # from pytorch_lightning.loggers import NeptuneLogger
#     # neptune_path = os.path.join(args.MYCHECKPOINT, 'neptune.npz')
#     # if args.neptune_id:
#     #     np.savez(
#     #         neptune_path,
#     #         project=args.neptune_proj,
#     #         id=args.neptune_id)
#     #     print(" >> newly create naptune.npz in test_pl.py")
#     #
#     # if os.path.exists(neptune_path):
#     #     neptune_info = np.load(neptune_path)
#     #
#     #     neptune_logger = NeptuneLogger(
#     #         api_key="<YOUR NEPTUNE>",
#     #         project=args.neptune_proj)
#     #     neptune_logger._run_short_id = str(neptune_info['id'])
#     #     print(">> re-use the neptune: id[%s]"%(neptune_info['id']))
#     # else:
#     #     neptune_logger = NeptuneLogger(
#     #         api_key="<YOUR NEPTUNE>",
#     #         project=args.neptune_proj)
#     #     print(">> start new neptune")
#     #
#     # neptune_logger.experiment["sys/tags"].add('test_pl.py')
#
#     # ------------
#     # model
#     # ------------
#     ckpts = sorted(glob.glob(os.path.join(args.MYCHECKPOINT, "*.ckpt")))
#     if len(ckpts)>1: ckpts = ckpts[:-1] # remove 'last.ckpt'
#     mIoU_val_best = -1.
#     filename_best = None
#     for ckpt in ckpts:
#         rootpath = '/'.join(ckpt.split('/')[:-1])
#         filename = ckpt.split('/')[-1]
#         mIoU_val = filename[:-5]
#         mIoU_val = float((mIoU_val.split('--')[-2])[9:])
#         if mIoU_val >= mIoU_val_best:
#             mIoU_val_best = mIoU_val
#             filename_best = filename
#     # ckpt_best = os.path.join(rootpath, filename_best)
#     ckpt_best = filename_best
#     args.load_model = ckpt_best
#     args.on_train = False
#     print('ckpt best. args.load_model=[{}]'.format(args.load_model))
#     assert args.load_model is not None, 'why did you come?'
#     model = get_model(args.model).load_from_checkpoint(
#         os.path.join(args.MYCHECKPOINT, args.load_model),
#         args=args,
#         strict=True) # args.strict_load
#
#     model.eval()
#     model.freeze()
#
#     # ------------
#     # trainer
#     # ------------
#     trainer = pl.Trainer(
#
#         accelerator="gpu",
#         strategy=DDPStrategy(find_unused_parameters=False), # 'ddp'
#         enable_progress_bar=False if 'NVIDIA' in args.computer else True,
#     )
#
#     # ------------
#     # test
#     # ------------
#     if args.dataset == 'loader_s3dis':
#         data_root = os.path.join(args.s3dis_root, 'trainval_fullarea')
#         data_list = sorted(os.listdir(data_root))
#         data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
#     else:
#         raise NotImplemented
#
#     ckpt_name = (args.load_model).split('/')[-1]
#     ckpt_name = ckpt_name[:-5] # remove '.ckpt'
#     save_folder = os.path.join(args.MYCHECKPOINT, 'test_results__%s'%((ckpt_name)))
#     os.makedirs(save_folder, exist_ok=True)
#
#     intersection_meter = AverageMeter()
#     union_meter = AverageMeter()
#     target_meter = AverageMeter()
#     pred_save, label_save = [], []
#
#     str_to_log = '<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<'
#     print(str_to_log)
#
#
#     for idx, item in enumerate(data_list):
#
#         pred_save_filename = \
#             '{}__epoch_{}npts{:09d}__size0p{:04d}__pred__test_pl__.npy'.format(
#                 item, model.current_epoch, args.eval_voxel_max, int(args.voxel_size*10000))
#         pred_save_path = os.path.join(save_folder, pred_save_filename)
#
#         label_save_filename = \
#             '{}__epoch_{}npts{:09d}__size0p{:04d}__label__test_pl__.npy'.format(
#                 item, model.current_epoch, args.eval_voxel_max, int(args.voxel_size*10000))
#         label_save_path = os.path.join(save_folder, label_save_filename)
#
#
#         if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
#             print('{}/{}: [{}], loaded pred and label.'.format(
#                 idx+1, len(data_list), item))
#             pred = np.load(pred_save_path)
#             label = np.load(label_save_path)
#
#         else:
#
#             if args.dataset == 'loader_s3dis':
#                 data_path = os.path.join(
#                     args.s3dis_root, 'trainval_fullarea', item+'.npy')
#                 data = np.load(data_path)
#                 label = data[:, 6] # coord, feat = data[:, :3], data[:, 3:6]
#                 mode_eval = 'test'
#
#             elif args.dataset == 'loader_scannet':
#                 filepath = os.path.join(
#                     args.scannet_semgseg_root,
#                     'val', item+'.pth')
#                 data = torch.load(filepath)
#                 label = np.array(data[2], dtype=np.int32)
#                 mode_eval = args.mode_eval # 'val'
#
#             elif args.dataset == 'loader_scannet_js':
#                 filepath = os.path.join(
#                     args.scannet_semgseg_root, 'train/%s.ply'%(item))
#                 plydata = PlyData.read(filepath)
#                 data = plydata.elements[0].data
#                 label = np.array(data['label'], dtype=np.int32)
#                 mode_eval = args.mode_eval # 'val'
#
#             else:
#                 raise NotImplemented
#
#
#             with torch.no_grad():
#                 model.pred = torch.zeros((label.size, args.classes)).cuda()
#
#             root = 'jilin_test/trainval_fullarea/'
#
#             TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=5,
#                                                                 block_points=4096)
#
#             dataset = get_dataset(args.dataset)
#             test_loader_kwargs = \
#                 {
#                     "batch_size": args.test_batch, # WRONG. Because of my stupid code. ,
#                     "num_workers": args.val_worker,
#                     "collate_fn": dataset.TestCollateFn,
#                     "pin_memory": False,
#                     "drop_last": False,
#                     "shuffle": False,
#                 }
#             #
#             # test_loader = torch.utils.data.DataLoader(TEST_DATASET_WHOLE_SCENE, **test_loader_kwargs)
#             # # test_loader = torch.utils.data.DataLoader(
#             # #     dataset.myImageFloder(args, mode=mode_eval, test_split=item),
#             # #     **test_loader_kwargs)
#             NUM_CLASSES = 5
#             label_values = range(NUM_CLASSES)
#             # BATCH_SIZE = args.batch_size
#             # NUM_POINT = args.num_point
#             BATCH_SIZE=args.test_batch
#             NUM_POINT=4096
#             num_votes=10
#             with torch.no_grad():
#                 scene_id = TEST_DATASET_WHOLE_SCENE.file_list
#                 scene_id = [x[:-4] for x in scene_id]
#                 num_batches = len(TEST_DATASET_WHOLE_SCENE)
#                 total_pred_class = [0 for _ in range(NUM_CLASSES)]
#                 total_seen_class = [0 for _ in range(NUM_CLASSES)]
#                 total_correct_class = [0 for _ in range(NUM_CLASSES)]
#                 total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
#                 recall = 0;
#                 recall_1 = []
#                 Precision_1 = []
#                 Precision = 0
#                 F_MC = []
#                 ACC = 0
#                 Confs = []
#                 for batch_idx in range(num_batches):
#                     print("visualize [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
#                     total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
#                     total_pred_class_tmp = [0 for _ in range(NUM_CLASSES)]
#                     total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
#                     total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
#                     whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
#                     whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
#                     vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
#                     for _ in tqdm(range(num_votes), total=num_votes):
#                         scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
#                         num_blocks = scene_data.shape[0]
#                         s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
#                         batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
#
#                         batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
#                         batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
#                         batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
#                         for sbatch in range(s_batch_num):
#                             start_idx = sbatch * BATCH_SIZE
#                             end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
#                             real_batch_size = end_idx - start_idx
#                             batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
#                             batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
#                             batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
#                             batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
#                             batch_data[:, :, 3:6] /= 1.0
#
#
#                             torch_data = torch.Tensor(batch_data)
#                             torch_data = torch_data.float().cuda()
#                             torch_data = torch_data.transpose(2, 1)
#                             test_loader = torch.utils.data.DataLoader(
#                                 (torch_data, batch_label,batch_point_index),
#                                     **test_loader_kwargs)
#                             trainer.test(model=model, dataloaders=test_loader, verbose=True)
#                             seg_pred = model.pred.max(1)[1].cpu().detach().numpy()
#
#
#                             batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
#
#                             vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
#                                                        batch_pred_label[0:real_batch_size, ...],
#                                                        batch_smpw[0:real_batch_size, ...])
#
#                     pred_label = np.argmax(vote_label_pool, 1)
#
#                     for l in range(NUM_CLASSES):
#                         total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
#                         total_pred_class_tmp[l] += np.sum((pred_label == l))
#                         total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
#                         total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
#                         total_seen_class[l] += total_seen_class_tmp[l]
#                         total_pred_class[l] += total_pred_class_tmp[l]
#                         total_correct_class[l] += total_correct_class_tmp[l]
#                         total_iou_deno_class[l] += total_iou_deno_class_tmp[l]
#                     iou_map = np.array(total_correct_class_tmp) / (
#                                 np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
#                     print(iou_map)
#
#     #         np.save(pred_save_path, pred)
#     #         np.save(label_save_path, label)
#     #         # end of it cond
#     #
#     #     # calculation 1: add per room predictions
#     #     intersection, union, target = \
#     #         intersectionAndUnion(pred, label, args.classes, args.ignore_label)
#     #     intersection_meter.update(intersection)
#     #     union_meter.update(union)
#     #     target_meter.update(target)
#     #
#     #     accuracy = sum(intersection) / (sum(target) + 1e-10)
#     #
#     #
#     #     str_to_log = \
#     #         'Test: [{:4d}/{:4d}]-npts[{npts:7d}/{:7d}] Accuracy[{accuracy:.4f}]'.format(
#     #             int(idx+1), len(data_list), int(label.size),
#     #             accuracy=accuracy,
#     #             npts=args.eval_voxel_max)
#     #     print(str_to_log)
#     #
#     #
#     #     pred_save.append(pred)
#     #     label_save.append(label)
#     #     # end of the for loop. (per-scene)
#     #
#     # # calculation 1
#     # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
#     # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
#     # mIoU1 = np.mean(iou_class)
#     # mAcc1 = np.mean(accuracy_class)
#     # allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
#     #
#     # # calculation 2
#     # intersection, union, target = \
#     #     intersectionAndUnion(
#     #         np.concatenate(pred_save),
#     #         np.concatenate(label_save),
#     #         args.classes,
#     #         args.ignore_label)
#     # iou_class = intersection / (union + 1e-10)
#     # accuracy_class = intersection / (target + 1e-10)
#     # mIoU = np.mean(iou_class)
#     # mAcc = np.mean(accuracy_class)
#     # allAcc = sum(intersection) / (sum(target) + 1e-10)
#     #
#     # str_to_log = \
#     #     'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
#     #         mIoU, mAcc, allAcc)
#     # print(str_to_log)
#     #
#     #
#     # str_to_log = \
#     #     'Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
#     #         mIoU1, mAcc1, allAcc1)
#     # print(str_to_log)
#     #
#     #
#     # if args.dataset == 'loader_s3dis':
#     #     names_path = os.path.join(args.s3dis_root, "list/s3dis_names.txt")
#     #     names = [line.rstrip('\n') for line in open(names_path)]
#     # elif args.dataset == 'loader_scannet':
#     #     names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
#     #             'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
#     #             'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
#     # else:
#     #     print("Please set labels' names.")
#     #     raise NotImplemented
#     #
#     # for i in range(args.classes):
#     #     str_to_log = \
#     #         'Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(
#     #             i, iou_class[i], accuracy_class[i], names[i])
#     #     print(str_to_log)
#     #
#     #
#     # str_to_log = '<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<'
#     # print(str_to_log)
#
#
# if __name__ == '__main__':
#     cli_main()