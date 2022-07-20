import numpy as np
import random
import yaml
import os
import cv2
from scipy.spatial import Delaunay

import pdb


def in_range(v, r):
    return (v >= r[0]) * (v < r[1])


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def compute_box_3d(center, size, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    return corners_3d.T


def rotate_along_z(pcds, theta):
    rotateMatrix = cv2.getRotationMatrix2D((0, 0), theta, 1.0)[:, :2].T
    pcds[:, :2] = pcds[:, :2].dot(rotateMatrix)
    return pcds


def random_f(r):
    return r[0] + (r[1] - r[0]) * random.random()


class SequenceCutPaste:
    def __init__(self, object_dir, paste_max_obj_num):
        self.object_dir = object_dir
        self.sub_dirs = ('other-vehicle', 'truck', 'car', 'motorcyclist', 'motorcycle', 'person', 'bicycle', 'bicyclist')
        '''
        other-vehicle: 7014
        bicycle: 4063
        motorcyclist: 530
        bicyclist: 1350
        motorcycle: 2774
        truck: 2514
        person: 6764
        '''
        self.velo_range_dic = {}
        self.velo_range_dic['other-vehicle'] = (-15, 15)
        self.velo_range_dic['truck'] = (-15, 15)
        self.velo_range_dic['car'] = (-15, 15)
        self.velo_range_dic['motorcyclist'] = (-8, 8)
        self.velo_range_dic['motorcycle'] = (-8, 8)
        self.velo_range_dic['person'] = (-3, 3)
        self.velo_range_dic['bicycle'] = (-8, 8)
        self.velo_range_dic['bicyclist'] = (-8, 8)
        
        self.sub_dirs_dic = {}
        for fp in self.sub_dirs:
            fpath = os.path.join(self.object_dir, fp)
            fname_list = [os.path.join(fpath, x) for x in os.listdir(fpath) if (x.endswith('.npz')) and (x.split('_')[0] != '08')]
            print('Load {0}: {1}'.format(fp, len(fname_list)))
            self.sub_dirs_dic[fp] = fname_list
        
        self.paste_max_obj_num = paste_max_obj_num
    
    def get_random_rotate_along_z_obj(self, pcds_obj, bbox_corners, theta):
        pcds_obj_result = rotate_along_z(pcds_obj, theta)
        bbox_corners_result = rotate_along_z(bbox_corners, theta)
        return pcds_obj_result, bbox_corners_result

    def get_fov(self, pcds_obj):
        x, y, z = pcds_obj[:, 0], pcds_obj[:, 1], pcds_obj[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        u_fov = (u.min(), u.max())
        phi_fov = (phi.min(), phi.max())
        theta_fov = (theta.min(), theta.max())
        return u_fov, phi_fov, theta_fov

    def occlusion_process(self, pcds, phi_fov, theta_fov):
        x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        fov_mask = in_range(phi, phi_fov) * in_range(theta, theta_fov)
        return fov_mask
    
    def make_sequential_obj(self, fname_npz, seq_num):
        npkl = np.load(fname_npz)

        pcds_obj = npkl['pcds']
        cate_id = int(npkl['cate_id'])
        semantic_cate = str(npkl['cate'])

        bbox_center = npkl['center']
        bbox_size = npkl['size'] * 1.05
        bbox_yaw = npkl['yaw']

        bbox_corners = compute_box_3d(bbox_center, bbox_size, bbox_yaw)

        velo = random_f(self.velo_range_dic[semantic_cate])
        velo_x = -1 * velo * np.sin(bbox_yaw)
        velo_y = velo * np.cos(bbox_yaw)

        pc_object_list = []
        for t in range(seq_num):
            # object pc
            pcds_obj_tmp = pcds_obj.copy()
            pcds_obj_tmp[:, 0] -= velo_x * t * 0.1
            pcds_obj_tmp[:, 1] -= velo_y * t * 0.1
            pcds_obj_tmp[:, :3] += np.random.normal(0, 0.001, size=(pcds_obj_tmp.shape[0], 3))

            # box corner
            bbox_corners_tmp = bbox_corners.copy()
            bbox_corners_tmp[:, 0] -= velo_x * t * 0.1
            bbox_corners_tmp[:, 1] -= velo_y * t * 0.1

            pc_object_list.append((pcds_obj_tmp, bbox_corners_tmp))
        
        return pc_object_list, np.abs(velo)
    
    def get_random_rotate_along_z_obj_list(self, pc_object_list, theta):
        result_pc_object_list = []
        for i in range(len(pc_object_list)):
            result_pc_object_list.append(self.get_random_rotate_along_z_obj(pc_object_list[i][0], pc_object_list[i][1], theta))
        return result_pc_object_list
    
    def valid_position(self, pcds, pcds_raw_label, pcds_obj):
        # get object fov
        u_fov, phi_fov, theta_fov = self.get_fov(pcds_obj)
        if (abs(u_fov[1] - u_fov[0]) < 8) and (abs(phi_fov[1] - phi_fov[0]) < 1) and (abs(theta_fov[1] - theta_fov[0]) < 1):
            # get valid fov
            fov_mask = self.occlusion_process(pcds, phi_fov, theta_fov)
            in_fov_obj_mask = in_range(pcds_raw_label[fov_mask], (10, 33)) + in_range(pcds_raw_label[fov_mask], (252, 260))
            if(in_fov_obj_mask.sum() < 3):
                return True, fov_mask
            else:
                return False, fov_mask
        else:
            return False, None
    
    def paste_single_obj(self, pcds_list, pcds_label_list, pcds_road_list, pcds_raw_label_list):
        '''
        Input:
            pcds_list, list of (N, 4), 4 -> x, y, z, intensity
            pcds_label_list, list of (N,)
            pcds_road_list, list of (M, 4)
            pcds_raw_label_list, list of (N,)
        Output:
            pcds_list, list of (N, 4), 4 -> x, y, z, intensity
            pcds_label_list, list of (N,)
            pcds_raw_label_list, list of (N,)
        '''
        cate = random.choice(self.sub_dirs)
        fname_npz = random.choice(self.sub_dirs_dic[cate])
        
        pc_object_list, obj_velo = self.make_sequential_obj(fname_npz, seq_num=len(pcds_list))
        motion_label = 0
        if obj_velo >= 1:
            motion_label = 2
        elif obj_velo < 0.3:
            motion_label = 1
        else:
            motion_label = 0
        
        if(len(pc_object_list[0][0]) < 10):
            return pcds_list, pcds_label_list, pcds_raw_label_list
        
        theta_list = np.arange(0, 360, 18).tolist()
        np.random.shuffle(theta_list)
        for theta in theta_list:
            # global rotate object
            pc_object_aug_list = self.get_random_rotate_along_z_obj_list(pc_object_list, theta)

            # current frame
            pcds_road = pcds_road_list[0]
            # get local road height
            valid_road_mask = in_hull(pcds_road[:, :2], pc_object_aug_list[0][1][:4, :2])
            pcds_local_road = pcds_road[valid_road_mask]
            if pcds_local_road.shape[0] > 5:
                road_mean_height = float(pcds_local_road[:, 2].mean())
                for ht in range(len(pc_object_aug_list)):
                    pc_object_aug_list[ht][0][:, 2] += road_mean_height - pc_object_aug_list[ht][0][:, 2].min()
            else:
                # object is not on road
                continue
            
            # get object list fov
            valid_position_list = [self.valid_position(pcds_list[ht], pcds_raw_label_list[ht], pc_object_aug_list[ht][0]) for ht in range(len(pc_object_aug_list))]
            valid_flag = True
            for ht in range(len(pc_object_aug_list)):
                valid_flag = valid_flag & valid_position_list[ht][0]
            
            if valid_flag:
                # add object back
                for ht in range(len(pc_object_aug_list)):
                    assert pcds_list[ht].shape[0] == pcds_label_list[ht].shape[0]
                    assert pcds_label_list[ht].shape[0] == pcds_raw_label_list[ht].shape[0]
                    _, fov_mask = valid_position_list[ht]

                    pcds_filter_ht = pcds_list[ht][~fov_mask]
                    pcds_label_filter_ht = pcds_label_list[ht][~fov_mask]
                    pcds_raw_label_filter_ht = pcds_raw_label_list[ht][~fov_mask]

                    pcds_obj_aug_ht = pc_object_aug_list[ht][0]
                    pcds_addobj_label_ht = np.full((pcds_obj_aug_ht.shape[0],), fill_value=motion_label, dtype=pcds_label_filter_ht.dtype)
                    pcds_addobj_raw_label_ht = np.full((pcds_obj_aug_ht.shape[0],), fill_value=30, dtype=pcds_raw_label_filter_ht.dtype)

                    pcds_list[ht] = np.concatenate((pcds_filter_ht, pcds_obj_aug_ht), axis=0)
                    pcds_label_list[ht] = np.concatenate((pcds_label_filter_ht, pcds_addobj_label_ht), axis=0)
                    pcds_raw_label_list[ht] = np.concatenate((pcds_raw_label_filter_ht, pcds_addobj_raw_label_ht), axis=0)
                break
            else:
                continue
        
        return pcds_list, pcds_label_list, pcds_raw_label_list
    
    def __call__(self, pcds_list, pcds_label_list, pcds_road_list, pcds_raw_label_list):
        paste_obj_num = random.randint(0, self.paste_max_obj_num)
        if paste_obj_num == 0:
            return pcds_list, pcds_label_list
        else:
            for i in range(paste_obj_num):
                pcds_list, pcds_label_list, pcds_raw_label_list = self.paste_single_obj(pcds_list, pcds_label_list, pcds_road_list, pcds_raw_label_list)
            
            return pcds_list, pcds_label_list