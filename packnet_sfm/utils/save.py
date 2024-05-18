# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from packnet_sfm.utils.image import write_image
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import prepare_dataset_prefix
from packnet_sfm.utils.util import get_colormap

# from packnet_sfm.utils.flowlib import flow_to_image

def save_depth(batch, output, args, dataset, save):
    """
    Save depth predictions in various ways

    Parameters
    ----------
    batch : dict
        Batch from dataloader
    output : dict
        Output from model
    args : tuple
        Step arguments
    dataset : CfgNode
        Dataset configuration
    save : CfgNode
        Save configuration
    """
    # If there is no save folder, don't save
    if save.folder is '':
        return

    # If we want to save
    if save.depth.rgb:
        # Retrieve useful tensors
        rgb = batch['rgb']
        pred_inv_depth = output['inv_depth']
        pred_inv_depth_pp = output['inv_depth_pp']
        error_maps = output['error_maps']
        gt_valid_maps = output['gt_valid_maps']
        far_close_maps = output['far_close_maps']
        os.makedirs(save.folder, exist_ok=True)
        torch.save({'error_maps': output['error_maps']['depth_pp_gt']}, save.folder + "/error_maps_%s.pt" % batch['filename'][0])

        # # Prepare path strings
        filename = batch['filename']
        # dataset_idx = 0 if len(args) == 1 else args[1]
        # save_path = os.path.join(save.folder, 'depth',
        #                          prepare_dataset_prefix(dataset, dataset_idx),
        #                          os.path.basename(save.pretrained).split('.')[0])
        # # Create folder
        # os.makedirs(save_path, exist_ok=True)

        # For each image in the batch
        length = rgb.shape[0]
        for i in range(length):
            # # Save numpy depth maps
            # if save.depth.npz:
            #     write_depth('{}/{}_depth.npz'.format(save_path, filename[i]),
            #                 depth=inv2depth(pred_inv_depth[i]),
            #                 intrinsics=batch['intrinsics'][i] if 'intrinsics' in batch else None)
            
            # # Save png depth maps
            # if save.depth.png:
            #     write_depth('{}/{}_depth.png'.format(save_path, filename[i]),
            #                 depth=inv2depth(pred_inv_depth[i]))

            # 绘制error map
            error_map = error_maps['depth_pp_gt'].cpu().detach()
            zeros = error_map==0.
            error_map_dense = get_colormap(error_map)
            error_map_dense = cv2.resize(error_map_dense, (rgb.shape[3], rgb.shape[2]))
            error_map = viz_inv_depth(error_map, filter_zeros=True) * 255
            error_map[zeros, :] = 255
            error_map = cv2.resize(error_map, (rgb.shape[3], rgb.shape[2]))

            # 绘制gt valid map
            gt_valid_map = gt_valid_maps['depth_pp_gt'].cpu().detach()
            zeros = gt_valid_map==0.
            gt_valid_map = viz_inv_depth(gt_valid_map, filter_zeros=True) * 255
            gt_valid_map[zeros, :] = 255
            gt_valid_map = cv2.resize(gt_valid_map, (rgb.shape[3], rgb.shape[2]))

            # 绘制远近图
            far_close_map = far_close_maps['depth_pp_gt'].cpu().detach()
            zeros = far_close_map==0.
            far_close_map_dense = get_colormap(far_close_map)
            far_close_map_dense = cv2.resize(far_close_map_dense, (rgb.shape[3], rgb.shape[2]))
            far_close_map = viz_inv_depth(far_close_map, filter_zeros=True) * 255
            far_close_map[zeros, :] = 255
            far_close_map = cv2.resize(far_close_map, (rgb.shape[3], rgb.shape[2]))[:, :, ::-1]

            # # 绘制error map对比图
            # error_cmp = torch.abs(error_maps['depth_pp_gt'] - cmp_error_map).cpu().detach()
            # error_cmp = get_colormap(error_cmp)
            # error_cmp = cv2.resize(error_cmp, (rgb.shape[3], rgb.shape[2]))[:, :, ::-1]

            compare = False
            network = 'optical' # 'semantic'
            baseNet = 'tree'
            if compare == False:
                # gt
                depth = batch['depth']
                inv_depth = 1. / depth[0]
                inv_depth[depth[0] == 0] = 0
                inv_depth = inv_depth[0].cpu().detach().numpy()
                depth = get_colormap(inv_depth)
                gt = cv2.resize(depth, (rgb.shape[3], rgb.shape[2]))
                if baseNet == 'packnet':
                    # 绘制预训练模型MR-K深度图: 远近图, error map, 预测深度后处理，gt
                    # inv_dep = viz_inv_depth(pred_inv_depth[i]) * 255
                    inv_dep_pp = viz_inv_depth(pred_inv_depth_pp[i]) * 255
                    rgb_viz_gt = np.concatenate([far_close_map_dense, error_map_dense, inv_dep_pp, gt], 0)[:, :, ::-1]
                    # cv2.imwrite('./data/MR-K_error/{}_rgb_viz_gt.png'.format(filename[i]), rgb_viz_gt)
                    cv2.imwrite('./data/MR-CS_error/{}_rgb_viz_gt.png'.format(filename[i]), rgb_viz_gt)
                elif baseNet == 'tree':
                    # 绘制结果图：原图，error map，预测深度, gt
                    img = rgb[i].permute(1, 2, 0).cpu().numpy() * 255
                    inv_dep = viz_inv_depth(pred_inv_depth[i]) * 255
                    inv_dep_pp = viz_inv_depth(pred_inv_depth_pp[i]) * 255
                    white_array = np.zeros((img.shape[0], img.shape[1], 3))
                    white_array[:, :, :] = 255
                    idx = batch['idx']
                    # text = '3HRNetMR\nidx: %d sqr: %.4f rmse: %.4f\na1: %.4f\nimg\nmonodepth2\nFeatDepth' \
                    #     % (idx, output['metrics']['depth_gt'][1], output['metrics']['depth_gt'][2], output['metrics']['depth_gt'][4])
                    text = '%s\nidx: %d sqr: %.4f rmse: %.4f\na1: %.4f' \
                        % (os.path.basename(save.folder), \
                           idx, output['metrics']['depth_gt'][1], \
                           output['metrics']['depth_gt'][2], output['metrics']['depth_gt'][4])
                    y0, dy = 50, 30
                    for i, txt in enumerate(text.split('\n')):
                        y = y0 + i*dy
                        cv2.putText(
                            img=white_array,
                            text=txt,
                            org=(40, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 0),
                            thickness=2
                        )

                    rgb_viz_gt = np.concatenate([white_array[:, :, ::-1], img[:, :, ::-1], \
                        error_map_dense[:, :, ::-1], inv_dep[:, :, ::-1], inv_dep_pp[:, :, ::-1], gt[:, :, ::-1]], 0)
                    cv2.imwrite(str(save.folder +'/' + str(idx.item()) + '_' + filename[0] + '.png'), \
                        rgb_viz_gt)
                    cv2.imwrite(str(save.folder +'/' + 'inv_dep_' + str(idx.item()) + '_' + filename[0] + '.png'), \
                        inv_dep[:, :, ::-1])
                    cv2.imwrite(str(save.folder +'/' + 'inv_dep_pp_' + str(idx.item()) + '_' + filename[0] + '.png'), \
                        inv_dep_pp[:, :, ::-1])

                    HiDNet = cv2.imread('data/HRAddNet_3_320_1024_HiDNet/{}_{}.png'.format(idx.item(), filename[0]))
                    rgb_viz_gt = np.concatenate([rgb_viz_gt, HiDNet], 1)
                    cv2.imwrite(str(save.folder +'/' + 'HiDNet_step_2_GMM_' + str(idx.item()) + '_' + filename[0] + '.png'), rgb_viz_gt)
            else:
                # 绘制结果图：原图，语义分割/光流图，error map，预测深度后处理
                img = rgb[i].permute(1, 2, 0).cpu().numpy() * 255
                if network == 'semantic':
                    colors = np.loadtxt('data/cityscapes/cityscapes_colors.txt').astype('uint8')
                    # # USPNet
                    # prediction = np.squeeze(np.copy(output['semantic']['fcn_outputs'][0].cpu().numpy()))
                    # # SGDepth
                    # prediction = np.squeeze(np.copy(torch.max(output['semantic'][0]['segmentation_logits', 0], dim=1)[1][0].cpu().numpy()))
                    # HRNet
                    prediction = np.squeeze(torch.argmax(output['semantic'], dim=1).cpu().detach().numpy())
                    gray = np.uint8(prediction)
                    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
                    color.putpalette(colors)
                    network_map = cv2.cvtColor(np.asarray(color.convert(palette=colors)), cv2.COLOR_RGB2BGR)[:, :, ::-1]
                    
                elif network == 'optical':
                    pass
                    # # 绘制光流图
                    # fwd_flow = output['fwd_flow'][1]
                    # fwd_flow = fwd_flow[0].permute(1, 2, 0).cpu().detach().numpy()
                    # # flow_map = flow_to_image(fwd_flow)[:, :, ::-1]
                    # # network_map = flow_map

                    # fwd_flow = np.sqrt(fwd_flow[:, :, 0]**2 + fwd_flow[:, :, 1]**2)
                    # network_map = get_colormap(fwd_flow, False)

                inv_dep_pp = viz_inv_depth(pred_inv_depth_pp[0]) * 255

                # output['loss_map']['perceptual_map']
                # left = np.concatenate([img, error_map, far_close_map_dense, error_map_dense, inv_dep_pp, output['loss_map']['ssim_map'], network_map], axis=0)[:, :, ::-1]
                left = np.concatenate([img, error_map, far_close_map_dense, error_map_dense, inv_dep_pp], axis=0)[:, :, ::-1]

                MR_K = cv2.imread('./data/MR-CS_error/{}_rgb_viz_gt.png'.format(filename[0]))
                white_array = np.zeros((img.shape[0], img.shape[1], 3))
                white_array[:, :, :] = 255
                cmp = torch.load('data/MR-CS_error/metrics.pt')
                cmp_metrics = cmp['metrics']
                idx = batch['idx']
                text = 'idx: %d sqr: %.4f %.4f %.4f\npredicted_error_map MR_error_map\npredicted_depth_pp  MR_depth_pp\nSGDepth50Pac        gt' \
                    % (idx, output['metrics']['depth_pp_gt'][1], cmp_metrics[idx, 1], output['metrics']['depth_pp_gt'][1] - cmp_metrics[idx, 1])
                # text = 'rgb\nopticalFlow         MR_error_map\npredicted_error_map MR_depth_pp\npredicted_depth_pp  gt'
                y0, dy = 50, 30
                for i, txt in enumerate(text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(
                        img=white_array,
                        text=txt,
                        org=(40, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=2
                    )
                # right = np.concatenate([white_array, far_close_map, MR_K, gt_valid_map[:, :, ::-1]], axis=0)
                right = np.concatenate([white_array, MR_K], axis=0)
                rgb_viz_gt = np.concatenate([left, right], axis=1)

                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                cv2.imwrite(str(save_folder_path +'/' + filename[0] + '.png'), rgb_viz_gt)
                print("Done with all pictures in: " + str(save_folder_path +'/' + filename[0] + '.png'))

            # # Save rgb images
            # if save.depth.rgb:
            #     rgb_i = rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255
            #     write_image('{}/{}_rgb.png'.format(save_path, filename[i]), rgb_i)
            # Save inverse depth visualizations
            if save.depth.viz:
                viz_i = viz_inv_depth(pred_inv_depth[i]) * 255
                write_image('{}/{}_viz.png'.format(save_path, filename[i]), viz_i)
    
    if save.depth.viz or save.depth.npz or save.depth.png:
        # Retrieve useful tensors
        rgb = batch['rgb']
        pred_inv_depth_pp = output['inv_depth_pp']

        # Prepare path strings
        filename = batch['filename']
        save_path = save.folder
        # Create folder
        os.makedirs(save_path, exist_ok=True)

        # For each image in the batch
        length = rgb.shape[0]
        for i in range(length):
            # Save numpy depth maps
            write_depth('{}/{}_depth.npz'.format(save_path, filename[i]),
                        depth=pred_inv_depth_pp[i],
                        GMM_prob=output['GMM_prob'][i].squeeze(),)

