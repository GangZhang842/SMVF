import argparse
import os
import importlib
import pdb

def parsing(line):
    line = line.strip()
    S = line.split(';')
    results_dic = {}

    results_dic['Epoch'] = S[0]
    for s in S[1:]:
        s = s.split(':')
        key = s[0].strip()
        value = float(s[1])
        results_dic[key] = value
    return results_dic


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    # set keywords
    fname_record_list = [os.path.join(save_path, x) for x in os.listdir(save_path) if x.startswith('record_')]
    
    key = 'moving iou'

    max_epoch = 0
    max_metric = 0
    for fname_record in fname_record_list:
        with open(fname_record, 'r') as f:
            for line in f:
                line_dic = parsing(line)
                if max_metric < line_dic[key]:
                    max_metric = line_dic[key]
                    max_epoch = line_dic['Epoch']
    
    print('Best Epoch: ', max_epoch, 'Best Metric: ', max_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)