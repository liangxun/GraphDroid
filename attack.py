import os
import torch
import argparse
import torch
import numpy as np
import random
import pickle as pkl
import json
import glob
from adv.utils import load_pretrain_model, load_data_for_adv, evl_index_for_adv
from adv.fgsm import FGSM
from adv.jsma import JSMA
from setting import *


map_attackers = {
    'fgsm': FGSM,
    'jsma': JSMA,
}


def adv_attack(model, data_loader, feat_data, max_bit, alg):
    attacker = map_attackers[alg](model, max_bit, torch.cuda.is_available())
    iter = 0
    r_codes = []
    for id, y in data_loader:
        if y.item() == 1:
            id = [i.item() for i in id.squeeze_(0)]
            y = y.squeeze_(0)
            init_x = feat_data[id]
            r_code, adv_x = attacker.attack(id, init_x, y)
            r_codes.append(r_code)
            iter += 1
            print('iter={}, r_code={}'.format(iter, r_code))
    evl_index_for_adv(r_codes, alg)
    return r_codes

def worker(args):
    alg, max_bit, model_file = args
    model, feat_data = load_pretrain_model(model_file)
    data_loader = load_data_for_adv()
    print("===START==attack algorithm:{} max_bit:{} ===START===".format(alg, max_bit))
    r_codes = adv_attack(model, data_loader, feat_data, max_bit, alg)
    report = {
        'alg': alg,
        'max_bit': max_bit,
        'model_file': model_file,
        'r_codes': r_codes,
    }
    print("===END==attack algorithm:{} max_bit:{} ===END===".format(alg, max_bit))
    return report


if __name__ == "__main__":
    target_models = list(map(lambda x: os.path.join(model_path, x), os.listdir(model_path)))
    target_models = list(filter(lambda x: x.split('.')[-1] != 'csv', target_models))

    commands = []
    for target_model_file in target_models:
        for max_bit in [10,20,30,40]:
            commands.append(('fgsm', max_bit, target_model_file))
        commands.append(('jsma', max_bit, target_model_file))
    
    print('{} commands will be performed'.format(len(commands)))
    for command in commands:
        print(command)

    import multiprocessing as mp
    pool = mp.Pool(processes=6)
    rets = pool.map(worker, commands)
    import json
    with open('attack.logger.json', 'w') as f:
        json.dump(rets, f, indent=4)

