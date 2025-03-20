from tools import _init_path
import argparse
import datetime
import glob
import os
import re
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import tqdm

from tools.eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models import build_network, model_fn_decorator
from torch.autograd import Variable
from pcdet.datasets.processor import data_processor

from sklearn.cluster import KMeans
import pandas as pd

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def clip_eta(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """
    grad_shape = grad.shape
    grad_shape_len = len(grad.shape)
    if grad_shape_len == 3:
        grad = grad.view(-1, 3)

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-36, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.sum(grad ** 2, red_ind, keepdim=True)
        optimal_perturbation = grad / torch.max(torch.sqrt(square), avoid_zero_div)
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    if grad_shape_len == 3:
        scaled_perturbation = scaled_perturbation.view(grad_shape)
    return scaled_perturbation

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, args, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    # define some hyper-parameters
    iter_eps = args.eps/30
    nb_iter = 40
    rand_init = True
    eps = args.eps # 0.3
    norm = 2 # np.inf 2
    decay_factor = 1
    clip_min = None
    clip_max = None
    model_func=model_fn_decorator()
    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    max_num_points_per_voxel = [x['MAX_POINTS_PER_VOXEL'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]
    max_num_voxels = [x['MAX_NUMBER_OF_VOXELS'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]['test']
    voxel_size = [x['VOXEL_SIZE'] for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x['NAME']=='transform_points_to_voxels'][0]

    num_point_features=4 if 'kitti' in args.cfg_file else 5
    voxel_generator = data_processor.VoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=point_cloud_range,
        num_point_features=num_point_features+1,
        max_num_points_per_voxel=max_num_points_per_voxel,
        max_num_voxels=max_num_voxels,
    )


    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        points_flatten = batch_dict[key].view(-1, num_point_features)
        points_sum = (points_flatten.abs()).sum(1)
        key_origin = points_flatten[points_sum!=0].clone()
        points_origin = key_origin.cpu().numpy()
        key_origin.requires_grad = False # important

        if rand_init:
            perturbation = torch.zeros_like(key_origin[:, :3]).uniform_(-eps, eps).cuda(key_origin.device)
            perturbation = clip_eta(perturbation, eps, norm)

            points_valid = copy.deepcopy(points_origin)
            points_valid[:, :3] = points_valid[:, :3] + perturbation.cpu().numpy()

            # re-voxelize

            points_valid[points_valid[:, 0]>=point_cloud_range[3], 0] = point_cloud_range[3] - 1e-6
            points_valid[points_valid[:, 1]>=point_cloud_range[4], 1] = point_cloud_range[4] - 1e-6
            points_valid[points_valid[:, 2]>=point_cloud_range[5], 2] = point_cloud_range[5] - 1e-6

            points_valid[points_valid[:, 0]<point_cloud_range[0], 0] = point_cloud_range[0]
            points_valid[points_valid[:, 1]<point_cloud_range[1], 1] = point_cloud_range[1]
            points_valid[points_valid[:, 2]<point_cloud_range[2], 2] = point_cloud_range[2]

            points_valid = np.concatenate([points_valid, np.arange(len(points_valid)).reshape(-1,1)], axis=1)
            voxels, coordinates, num_points = voxel_generator.generate(points_valid)

            batch_dict['voxels'] = torch.from_numpy(voxels[:, :, :num_point_features]).float().cuda(key_origin.device)
            pad_batch_indexs = np.zeros((len(voxels),1))
            coordinates = np.concatenate([pad_batch_indexs, coordinates], axis=1)
            batch_dict['voxel_coords'] = torch.from_numpy(coordinates).float().cuda(key_origin.device)
            batch_dict['voxel_num_points'] = torch.from_numpy(num_points).float().cuda(key_origin.device)
        else:
            voxel_points_index = torch.zeros(batch_dict['voxels'].shape[0] * batch_dict['voxels'].shape[1], device=key_origin.device)
            valid_points_index = torch.arange(len(key_origin), device=key_origin.device)
            try:
                voxel_points_index[torch.sum(batch_dict['voxels'].abs(), dim=2).flatten().nonzero().flatten()] = valid_points_index.float()
            except:
                import pdb;pdb.set_trace()
            voxels = torch.cat([batch_dict[key], voxel_points_index.view(batch_dict['voxels'].shape[0], batch_dict['voxels'].shape[1], 1)], axis=2)
            voxels = voxels.cpu().numpy()
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

        batch_dict[key].requires_grad = True
        
        g = torch.zeros_like(batch_dict[key][:, :, :3]).to(key_origin.device)

        for i in range(nb_iter):
            for cur_module in model.module_list:
                batch_dict = cur_module(batch_dict)
            loss, tb_dict, _ = model.get_training_loss()

            model.zero_grad()
            batch_dict[key].retain_grad()
            loss.backward(retain_graph=True)
            grad = batch_dict[key].grad.data

            grad[batch_dict[key]==0] = 0
            grad = grad[:, :, :3]

            if 'second' in args.cfg_file or 'voxel_rcnn' in args.cfg_file or 'PartA2' in args.cfg_file:
                grad = - grad

            perturbation = clip_eta(grad, iter_eps, norm)


            batch_dict[key].requires_grad = False
            batch_dict[key][:, :, :3] = batch_dict[key][:, :, :3] + perturbation

            batch_dict[key].requires_grad = True

            for k in ['batch_index', 'point_cls_scores', 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized', 'rois', 'roi_scores', 'roi_labels', 'has_class_labels']:
                batch_dict.pop(k, None) # adhoc

            for k in ['voxel_features', 'encoded_spconv_tensor', 'encoded_spconv_tensor_stride', 'multi_scale_3d_features', 'multi_scale_3d_strides', 'spatial_features', 'spatial_features_stride', 'spatial_features_2d']:
                batch_dict.pop(k, None)

            # If clipping is needed, reset all values outside of [clip_min, clip_max]
            if (clip_min is not None) or (clip_max is not None):
                if clip_min is None or clip_max is None:
                    raise ValueError(
                        "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                    )

        if args.defense:
            # 1) get points
            voxels_flatten = batch_dict[key].view(-1, 4)
            points = voxels_flatten[voxels_flatten.sum(1)!=0]
            # 2) voxelization
            voxel_generator_defense = data_processor.VoxelGeneratorWrapper(
                vsize_xyz=voxel_size,
                coors_range_xyz=point_cloud_range,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels,
            )
            points = points.cpu().detach().numpy()

            #gaussian:
            # noise = np.random.randn(*points[:, :3].shape) * (0.05**0.5)
            noise = np.random.normal(0, 0.01, points[:, :3].shape)
            points[:, :3] += noise

            try:
                voxels, coordinates, num_points = voxel_generator_defense.generate(points)
            except:
                import pdb;pdb.set_trace()

            coordinates = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)
            voxels = torch.from_numpy(voxels).float().cuda(key_origin.device)
            coordinates = torch.from_numpy(coordinates).float().cuda(key_origin.device)
            num_points = torch.from_numpy(num_points).float().cuda(key_origin.device)
            batch_dict['voxels'] = voxels
            batch_dict['voxel_coords'] = coordinates
            batch_dict['voxel_num_points'] = num_points

        if args.save_points:
            attack_type = os.path.basename(__file__).split('.')[0]
            frame_id = batch_dict['frame_id'][0]
            model_name = os.path.basename(args.cfg_file).split('.')[0]
            output_dir = f'./output/kitti_models/{model_name}/{attack_type}_PGD_{eps}/'
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'./output/kitti_models/{model_name}/{attack_type}_PGD_{eps}/{frame_id}.bin'
            save_points = points_valid[:, :4]

            # print('attempt to save', type(save_points))
            # save_points.tofile(output_path)

            with open(output_path, 'w') as f:
                save_points.tofile(f)
                # how to load: obj_points = np.fromfile(str(output_path), dtype=np.float32).reshape(-1 ,4)

        model.eval()
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None,
            flip=False, rotate=False, scale=False
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))


    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eps', type=float, default=0.03, help='max_shift default 0.03m')
    parser.add_argument('--defense', action='store_true', default=False, help='')
    parser.add_argument('--save_points', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    return args, cfg

def main():
    args, cfg = parse_config()
    dist_test = False
    total_gpus = 1

    assert 1 % total_gpus == 0, 'Batch size should match the number of gpus'
    batch_size = 1 // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'default'
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']

    eval_output_dir = eval_output_dir / 'default'

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_test, workers=4, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    ## rewrite the process of evaluation
    eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, args, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


if __name__ == '__main__':
    main()





