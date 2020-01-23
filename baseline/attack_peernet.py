import sys
import os
import torch 
from adv.alg_peernet import FGSM, JSMA
from utils import load_data_for_adv, load_pretrain_model, evl_index_for_adv, load_train_dataset
from config import *


is_cuda = torch.cuda.is_available()
if is_cuda:
    torch.cuda.set_device(cuda_device_id)

map_attackers = {
    'fgsm': FGSM,
    'jsma': JSMA,
}


def adv_attack(model, train_dataset, data_loader, max_bit, alg):

    attacker = map_attackers[alg](model, train_dataset, max_bit, is_cuda=is_cuda)

    iter = 0
    r_codes = []
    for x, y in data_loader:
        if y.item() == 1:
            r_code, adv_x = attacker.attack(x, y)
            r_codes.append(r_code)
            print('iter={}, r_code={}'.format(iter, r_code))
            iter += 1
    evl_index_for_adv(r_codes, alg)
    return r_codes


def worker(args):
    alg, max_bit, model_file = args
    model = load_pretrain_model(model_file)
    data_loader = load_data_for_adv(os.path.join(data_dir, 'baseline_dataset.pkl'))
    train_samples = load_train_dataset(os.path.join(data_dir, 'baseline_dataset.pkl'))

    print("=============attack algorithm:{} max_bit:{} ======START=========".format(alg, max_bit))
    r_codes = adv_attack(model, train_samples, data_loader, max_bit, alg)
    report = {
        'alg': alg,
        'max_bit': max_bit,
        'model_file': model_file,
        'r_codes': r_codes,
    }
    print("=============attack algorithm:{} max_bit:{} ======END=========".format(alg, max_bit))
    return report


if __name__ == '__main__':
    import json
    commands = []

    target_models = [os.path.join(model_save_dir, 'PeerNet_f10.9932_2020-01-07.pt')]
    print(target_models)

    rets = []
    command = ('jsma', 40, target_models[0])
    rets.append(worker(command))
    with open('attack_peernet_jsma.logger.json', 'w') as f:
        json.dump(rets, f, indent=4)

