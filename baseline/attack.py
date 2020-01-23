import sys
import os
from adv.fgsm import FGSM
from adv.jsma import JSMA
from utils import load_data_for_adv, load_pretrain_model, evl_index_for_adv
from config import *


map_attackers = {
    'fgsm': FGSM,
    'jsma': JSMA,
}


def adv_attack(model, data_loader, max_bit, alg):
    attacker = map_attackers[alg](model, max_bit)

    iter = 0
    r_codes = []
    for x, y in data_loader:
        if y.item() == 1:
            r_code, adv_x = attacker.attack(x, y)
            r_codes.append(r_code)
            iter += 1
            # if iter > 5:
            #     break
    evl_index_for_adv(r_codes, alg)
    return r_codes


def worker(args):
    alg, max_bit, model_file = args
    model = load_pretrain_model(model_file)
    data_loader = load_data_for_adv(os.path.join(data_dir, 'baseline_dataset.pkl'))
    print("=============attack algorithm:{} max_bit:{} ======START=========".format(alg, max_bit))
    r_codes = adv_attack(model, data_loader, max_bit, alg)
    report = {
        'alg': alg,
        'max_bit': max_bit,
        'model_file': model_file,
        'r_codes': r_codes,
    }
    print("=============attack algorithm:{} max_bit:{} ======END=========".format(alg, max_bit))
    return report


if __name__ == '__main__':
    commands = []

    target_models = list(map(lambda x: os.path.join(model_save_dir, x), os.listdir(model_save_dir)))
    print(target_models)

    for target_model_file in target_models:
        for max_bit in [10,20,30,40]:
            commands.append(('fgsm', max_bit, target_model_file))
        commands.append(('jsma', max_bit, target_model_file))

    print(commands)
    print(len(commands))

    import multiprocessing as mp
    pool = mp.Pool(processes=6)
    rets = pool.map(worker, commands)
    # import json
    # with open('attack.logger.json', 'w') as f:
    #     json.dump(rets, f, indent=4)
