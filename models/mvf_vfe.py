import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbone, bird_view, range_view
from networks.backbone import get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax

import yaml
import copy
import pdb


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / (pModel.Voxel.bev_shape[0])
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / (pModel.Voxel.bev_shape[1])
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / (pModel.Voxel.bev_shape[2])

        self.point_feat_out_channels = pModel.point_feat_out_channels

        self.build_network()
        self.build_loss()

    def build_loss(self):
        self.criterion_seg_cate = None
        print("Loss mode: {}".format(self.pModel.loss_mode))
        if self.pModel.loss_mode == 'ce':
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == 'ohem':
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=4.0, ignore_index=0)
        elif self.pModel.loss_mode == 'wce':
            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open('datasets/semantic-kitti.yaml', 'r') as f:
                task_cfg = yaml.load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg['learning_map'][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0
            print("Loss weights from content: ", loss_w)
            self.criterion_seg_cate = nn.CrossEntropyLoss(weight=loss_w)
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

    def build_network(self):
        # build network
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point = self.pModel.BEVParam.bev_grid2point

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point = self.pModel.RVParam.rv_grid2point

        fusion_mode = self.pModel.fusion_mode

        bev_context_layer[0] = self.pModel.seq_num * rv_context_layer[0]

        # network
        self.point_pre = backbone.PointNetStacker(7, rv_context_layer[0], pre_bn=True, stack_num=2)
        self.bev_net = bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers, use_att=True)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, use_att=True)
        self.bev_grid2point = get_module(bev_grid2point, in_dim=self.bev_net.out_channels)
        self.rv_grid2point = get_module(rv_grid2point, in_dim=self.rv_net.out_channels)

        point_fusion_channels = (rv_context_layer[0], self.bev_net.out_channels, self.rv_net.out_channels)
        self.point_post = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)

        self.pred_layer = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)
        self.pred_bev_layer = nn.Sequential(
            nn.Conv2d(rv_context_layer[0] + self.bev_net.out_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            backbone.PredBranch(64, self.pModel.class_num)
        )

    def stage_forward(self, point_feat, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            point_feat (BS, T, C, N, 1)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            point_feat_out (BS, C1, N, 1)
        '''
        BS, T, C, N, _ = point_feat.shape

        pcds_cood_cur = pcds_coord[:, 0, :, :2].contiguous()
        pcds_sphere_coord_cur = pcds_sphere_coord[:, 0].contiguous()

        # BEV
        point_feat_tmp = self.point_pre(point_feat.view(BS*T, C, N, 1))
        bev_input = VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord.view(BS*T, N, 3, 1)[:, :, :2].contiguous(), output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0)) #(BS*T, C, H, W)
        bev_input = bev_input.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])
        bev_feat = self.bev_net(bev_input)
        point_bev_feat = self.bev_grid2point(bev_feat, pcds_cood_cur)

        # range-view
        point_feat_tmp_cur = point_feat_tmp.view(BS, T, -1, N, 1)[:, 0].contiguous()
        rv_input = VoxelMaxPool(pcds_feat=point_feat_tmp_cur, pcds_ind=pcds_sphere_coord_cur, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        rv_feat = self.rv_net(rv_input)
        point_rv_feat = self.rv_grid2point(rv_feat, pcds_sphere_coord_cur)

        # merge multi-view
        point_feat_out = self.point_post(point_feat_tmp_cur, point_bev_feat, point_rv_feat)
        point_feat_out_bev = torch.cat((point_feat_tmp_cur, point_bev_feat), dim=1)

        # pred
        pred_cls = self.pred_layer(point_feat_out).float()
        pred_bev_cls = self.pred_bev_layer(point_feat_out_bev).float()
        return pred_cls, pred_bev_cls

    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        '''
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss

    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord, pcds_coord_raw (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        Output:
            loss
        '''
        pred_cls, pred_bev_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        pred_cls_raw, pred_bev_cls_raw = self.stage_forward(pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)

        loss1 = self.criterion_seg_cate(pred_cls, pcds_target) + 2 * lovasz_softmax(pred_cls, pcds_target, ignore=0)
        loss2 = self.criterion_seg_cate(pred_cls_raw, pcds_target) + 2 * lovasz_softmax(pred_cls_raw, pcds_target, ignore=0)
        loss3 = self.consistency_loss_l1(pred_cls, pred_cls_raw)

        loss_bev1 = self.criterion_seg_cate(pred_bev_cls, pcds_target) + 2 * lovasz_softmax(pred_bev_cls, pcds_target, ignore=0)
        loss_bev2 = self.criterion_seg_cate(pred_bev_cls_raw, pcds_target) + 2 * lovasz_softmax(pred_bev_cls_raw, pcds_target, ignore=0)

        loss = 0.5 * (loss1 + loss2) + loss3 + 0.5 * (loss_bev1 + loss_bev2)
        return loss

    def infer_val(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target):
        '''
        Input:
            pcds_xyzi (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        Output:
            pred_cls, (BS, C, N, 1)
        '''
        pred_cls, pred_bev_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        return pred_cls, pcds_target

    def infer_test(self, point_feat, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            point_feat (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls, (BS, C, N, 1)
        '''
        BS, T, C, N, _ = point_feat.shape

        pcds_cood_cur = pcds_coord[:, 0, :, :2].contiguous()
        pcds_sphere_coord_cur = pcds_sphere_coord[:, 0].contiguous()

        # BEV
        point_feat_tmp = self.point_pre(point_feat.view(BS*T, C, N, 1))
        bev_input = VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord.view(BS*T, N, 3, 1)[:, :, :2].contiguous(), output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0)) #(BS*T, C, H, W)
        bev_input = bev_input.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])
        bev_feat = self.bev_net(bev_input)
        point_bev_feat = self.bev_grid2point(bev_feat, pcds_cood_cur)

        # range-view
        point_feat_tmp_cur = point_feat_tmp.view(BS, T, -1, N, 1)[:, 0].contiguous()
        rv_input = VoxelMaxPool(pcds_feat=point_feat_tmp_cur, pcds_ind=pcds_sphere_coord_cur, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        rv_feat = self.rv_net(rv_input)
        point_rv_feat = self.rv_grid2point(rv_feat, pcds_sphere_coord_cur)

        # merge multi-view
        point_feat_out = self.point_post(point_feat_tmp_cur, point_bev_feat, point_rv_feat)

        # pred
        pred_cls = self.pred_layer(point_feat_out).float()
        return pred_cls