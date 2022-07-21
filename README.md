# **Sequential Multi-View Fusion Network for Fast LiDAR Point Motion Estimation**
![teaser](./imgs/overview-0.png)

Official code for SMVF

> **Sequential Multi-View Fusion Network for Fast LiDAR Point Motion Estimation**,
> Gang Zhang, Xiaoyan Li, Zhenhua Wang.
> *Accepted by ECCV2022*

## NEWS

- [2022-07-03] SMVF is accepted by ECCV 2022
- [2022-03-07] SMVF achieves 1st place in [SemanticKITTI Moving Object Segmentation leaderboard](https://competitions.codalab.org/competitions/28894#results).
![teaser](./imgs/smvf_semkitti-0.png)

#### 1 Dependency

```bash
CUDA>=10.1
Pytorch>=1.5.1
PyYAML@5.4.1
scipy@1.3.1
```

#### 2 Training Process

##### 2.1 Installation

```bash
cd deep_point
python setup.py install
```

##### 2.2 Prepare Dataset

Please download the [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#overview) dataset to the folder `data` and the structure of the folder should look like:

```
./
├── 
├── ...
└── dataset/
    ├──sequences
        ├── 00/         
        │   ├── velodyne/
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	        └── ...
```

##### 2.3 Training Script

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --config config_smvf_sgd_ohem_vfe_k2_fp16_48epoch.py
```

#### 3 Evaluate Process

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 evaluate.py --config config_smvf_sgd_ohem_vfe_k2_fp16_48epoch.py --start_epoch 0 --end_epoch 47
```
