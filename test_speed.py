import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    #define dataloader
    test_dataset = eval('datasets.{}.DataloadTest'.format(pDataset.Val.data_src))(pDataset.Val)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    test_loader = iter(test_loader)

    #define model
    model = eval(pModel.prefix)(pModel)
    model.eval()
    model.cuda()

    pcds_xyzi, pcds_coord, pcds_sphere_coord, valid_mask_list, pad_length_list, meta_list_raw = test_loader.next()

    pcds_xyzi = pcds_xyzi[0, [0]].contiguous().cuda()
    pcds_coord = pcds_coord[0, [0]].contiguous().cuda()
    pcds_sphere_coord = pcds_sphere_coord[0, [0]].contiguous().cuda()
    pdb.set_trace()

    time_cost = []
    with torch.no_grad():
        for i in range(1000):
            start = time.time()
            pred_cls = model.infer_test(pcds_xyzi, pcds_coord, pcds_sphere_coord)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end - start)

    print('Time: ', np.array(time_cost[20:]).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)