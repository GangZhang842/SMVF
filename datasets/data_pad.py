import torch

import PIL.Image as Im
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import numpy.linalg as lg

import yaml
import random
import json
from . import utils, copy_paste
import os


def make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, Voxel):
    # make point feat
    x = pcds_xyzi[:, 0].copy()
    y = pcds_xyzi[:, 1].copy()
    z = pcds_xyzi[:, 2].copy()
    intensity = pcds_xyzi[:, 3].copy()

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12

    # grid diff
    diff_x = pcds_coord[:, 0] - np.floor(pcds_coord[:, 0])
    diff_y = pcds_coord[:, 1] - np.floor(pcds_coord[:, 1])
    diff_z = pcds_coord[:, 2] - np.floor(pcds_coord[:, 2])

    # sphere diff
    phi_range_radian = (-np.pi, np.pi)
    theta_range_radian = (Voxel.RV_theta[0] * np.pi / 180.0, Voxel.RV_theta[1] * np.pi / 180.0)

    phi = phi_range_radian[1] - np.arctan2(x, y)
    theta = theta_range_radian[1] - np.arcsin(z / dist)

    diff_phi = pcds_sphere_coord[:, 0] - np.floor(pcds_sphere_coord[:, 0])
    diff_theta = pcds_sphere_coord[:, 1] - np.floor(pcds_sphere_coord[:, 1])

    point_feat = np.stack((x, y, z, intensity, dist, diff_x, diff_y), axis=-1)
    return point_feat


# define the class of dataloader
class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/semantic-kitti.yaml', 'r') as f:
            self.task_cfg = yaml.load(f)
        
        self.cp_aug = None
        if config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste.SequenceCutPaste(config.CopyPasteAug.ObjBackDir, config.CopyPasteAug.paste_max_obj_num)
        
        self.aug = utils.DataAugment(noise_mean=config.AugParam.noise_mean,
                        noise_std=config.AugParam.noise_std,
                        theta_range=config.AugParam.theta_range,
                        shift_range=config.AugParam.shift_range,
                        size_range=config.AugParam.size_range)
        
        self.aug_raw = utils.DataAugment(noise_mean=0,
                        noise_std=0,
                        theta_range=(0, 0),
                        shift_range=((0, 0), (0, 0), (0, 0)),
                        size_range=(1, 1))
        
        seq_num = config.seq_num
        # add training data
        seq_split = [str(i).rjust(2, '0') for i in self.task_cfg['split']['train']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, 'velodyne')
            fpath_label = os.path.join(fpath, 'labels')

            fname_calib = os.path.join(fpath, 'calib.txt')
            fname_pose = os.path.join(fpath, 'poses.txt')

            calib = utils.parse_calibration(fname_calib)
            poses_list = utils.parse_poses(fname_pose, calib)
            for i in range(len(poses_list)):
                meta_list = []
                meta_list_raw = []
                current_pose_inv = np.linalg.inv(poses_list[i])
                if (i < (seq_num - 1)):
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                else:
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                    
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                
                self.flist.append((meta_list, meta_list_raw))
        
        print('Training Samples: ', len(self.flist))
    
    def form_batch(self, pcds_total):
        #augment pcds
        pcds_total = self.aug(pcds_total)

        N = pcds_total.shape[0] // self.config.seq_num
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord
    
    def form_batch_raw(self, pcds_total):
        #augment pcds
        pcds_total = self.aug_raw(pcds_total)

        N = pcds_total.shape[0] // self.config.seq_num
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord
    
    def form_seq(self, meta_list):
        pc_list = []
        pc_label_list = []
        pc_raw_label_list = []
        pc_road_list = []
        for ht in range(self.config.seq_num):
            fname_pcd, fname_label, pose_diff, _, _ = meta_list[ht]
            # load pcd
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)
            pc_list.append(pcds_ht)

            # load label
            pcds_label = np.fromfile(fname_label, dtype=np.uint32)
            pcds_label = pcds_label.reshape((-1))
            sem_label = pcds_label & 0xFFFF
            inst_label = pcds_label >> 16

            pc_road_list.append(pcds_ht[sem_label == 40])

            pcds_label_use = utils.relabel(sem_label, self.task_cfg['learning_map'])
            pc_label_list.append(pcds_label_use)
            pc_raw_label_list.append(sem_label)
        
        return pc_list, pc_label_list, pc_road_list, pc_raw_label_list
    
    def __getitem__(self, index):
        meta_list, meta_list_raw = self.flist[index]
        # load history pcds
        pc_list = None
        pc_label_list = None
        pc_road_list = None
        if random.random() > 0.5:
            pc_list, pc_label_list, pc_road_list, pc_raw_label_list = self.form_seq(meta_list)
        else:
            pc_list, pc_label_list, pc_road_list, pc_raw_label_list = self.form_seq(meta_list_raw)
        
        # copy-paste
        if self.cp_aug is not None:
            pc_list, pc_label_list = self.cp_aug(pc_list, pc_label_list, pc_road_list, pc_raw_label_list)
        
        # filter
        for ht in range(len(pc_list)):
            valid_mask_ht = utils.filter_pcds_mask(pc_list[ht],
                                                range_x=self.Voxel.range_x,
                                                range_y=self.Voxel.range_y,
                                                range_z=self.Voxel.range_z)
            
            pc_list[ht] = pc_list[ht][valid_mask_ht]
            pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]
        
        # max length pad
        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length > 0
            pc_list[ht] = np.pad(pc_list[ht], ((0, pad_length), (0, 0)), 'constant', constant_values=-1000)
            pc_list[ht][-pad_length:, 2] = -4000

            pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), 'constant', constant_values=0)
        
        pc_list = np.concatenate(pc_list, axis=0)
        pcds_target = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)

        # preprocess
        pcds_xyzi, pcds_coord, pcds_sphere_coord = self.form_batch(pc_list.copy())
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw = self.form_batch_raw(pc_list.copy())
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, meta_list_raw
    
    def __len__(self):
        return len(self.flist)


# define the class of dataloader
class DataloadVal(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/semantic-kitti.yaml', 'r') as f:
            self.task_cfg = yaml.load(f)
        
        seq_num = config.seq_num
        # add training data
        seq_split = [str(i).rjust(2, '0') for i in self.task_cfg['split']['valid']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcd = os.path.join(fpath, 'velodyne')
            fpath_label = os.path.join(fpath, 'labels')

            fname_calib = os.path.join(fpath, 'calib.txt')
            fname_pose = os.path.join(fpath, 'poses.txt')

            calib = utils.parse_calibration(fname_calib)
            poses_list = utils.parse_poses(fname_pose, calib)
            for i in range(len(poses_list)):
                meta_list = []
                meta_list_raw = []
                current_pose_inv = np.linalg.inv(poses_list[i])
                if (i < (seq_num - 1)):
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                elif i > (len(poses_list) - seq_num):
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                else:
                    # forward
                    for ht in range(seq_num):
                        file_id = str(i - ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i - ht])
                        meta_list_raw.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                    
                    # backward
                    for ht in range(seq_num):
                        file_id = str(i + ht).rjust(6, '0')
                        fname_pcd = os.path.join(fpath_pcd, '{}.bin'.format(file_id))
                        fname_label = os.path.join(fpath_label, '{}.label'.format(file_id))
                        
                        pose_diff = current_pose_inv.dot(poses_list[i + ht])
                        meta_list.append((fname_pcd, fname_label, pose_diff, seq_id, file_id))
                
                self.flist.append((meta_list, meta_list_raw))
        
        print('Evaluation Samples: ', len(self.flist))
    
    def form_batch(self, pcds_total):
        N = pcds_total.shape[0] // self.config.seq_num
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_xyzi = pcds_xyzi.permute(0, 2, 1, 3).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32)).view(self.config.seq_num, N, -1, 1)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord
    
    def form_batch_tta(self, pcds_total):
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_sphere_coord_list = []
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pcds_tmp = pcds_total.copy()
                pcds_tmp[:, 0] *= x_sign
                pcds_tmp[:, 1] *= y_sign
                pcds_xyzi, pcds_coord, pcds_sphere_coord = self.form_batch(pcds_tmp)

                pcds_xyzi_list.append(pcds_xyzi)
                pcds_coord_list.append(pcds_coord)
                pcds_sphere_coord_list.append(pcds_sphere_coord)
        
        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_sphere_coord = torch.stack(pcds_sphere_coord_list, dim=0)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord
    
    def form_seq(self, meta_list):
        pc_list = []
        pc_label_list = []
        for ht in range(self.config.seq_num):
            fname_pcd, fname_label, pose_diff, _, _ = meta_list[ht]
            # load pcd
            pcds_tmp = np.fromfile(fname_pcd, dtype=np.float32).reshape((-1, 4))
            pcds_ht = utils.Trans(pcds_tmp, pose_diff)
            pc_list.append(pcds_ht)

            # load label
            pcds_label = np.fromfile(fname_label, dtype=np.uint32)
            pcds_label = pcds_label.reshape((-1))
            sem_label = pcds_label & 0xFFFF
            inst_label = pcds_label >> 16

            pcds_label_use = utils.relabel(sem_label, self.task_cfg['learning_map'])
            pc_label_list.append(pcds_label_use)
        
        return pc_list, pc_label_list
    
    def __getitem__(self, index):
        meta_list, meta_list_raw = self.flist[index]
        
        # load history pcds
        pc_list, pc_label_list = self.form_seq(meta_list_raw)

        valid_mask_list = []
        # filter
        for ht in range(len(pc_list)):
            valid_mask_ht = utils.filter_pcds_mask(pc_list[ht],
                                                range_x=self.Voxel.range_x,
                                                range_y=self.Voxel.range_y,
                                                range_z=self.Voxel.range_z)
            
            pc_list[ht] = pc_list[ht][valid_mask_ht]
            pc_label_list[ht] = pc_label_list[ht][valid_mask_ht]
            valid_mask_list.append(valid_mask_ht)
        
        pad_length_list = []
        # max length pad
        for ht in range(len(pc_list)):
            pad_length = self.frame_point_num - pc_list[ht].shape[0]
            assert pad_length > 0
            pc_list[ht] = np.pad(pc_list[ht], ((0, pad_length), (0, 0)), 'constant', constant_values=-1000)
            pc_list[ht][-pad_length:, 2] = -4000

            pc_label_list[ht] = np.pad(pc_label_list[ht], ((0, pad_length),), 'constant', constant_values=0)
            pad_length_list.append(pad_length)
        
        pc_list = np.concatenate(pc_list, axis=0)
        pcds_target = torch.LongTensor(pc_label_list[0].astype(np.long)).unsqueeze(-1)
        
        # preprocess
        pcds_xyzi, pcds_coord, pcds_sphere_coord = self.form_batch_tta(pc_list.copy())
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, valid_mask_list, pad_length_list, meta_list_raw

    def __len__(self):
        return len(self.flist)