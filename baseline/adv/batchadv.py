"""
generate adversaral samples for a batch of inputs
"""

import torch
from adv.jsma import JSMA
from adv.fgsm import FGSM
import random

map_attackers = {
    'fgsm': FGSM,
    'jsma': JSMA,
}

def gen_adv_for_batch(model, xs, ys, alg, max_bit, is_cuda):
    attacker = map_attackers[alg](model, max_bit, is_cuda)
    adv_xs = []
    for i in range(xs.shape[0]):
        _, adv_x = attacker.attack(xs[i].unsqueeze(0), ys[i].unsqueeze(0))
        adv_xs.append(adv_x)
    adv_xs = torch.cat(adv_xs, dim=0)

    all_x = torch.cat([xs, adv_xs], dim=0)
    all_y = ys.repeat(2)

    return all_x, all_y


    