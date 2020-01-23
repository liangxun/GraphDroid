"""
分析r_codes结果
"""
import os
import json
import numpy as np


# report = {
#         'alg': alg,
#         'max_bit': max_bit,
#         'model_file': model_file,
#         'r_codes': r_codes,
#     }


def compute_index(cnt, rc_0, rc_n, rc_p):
    assert cnt == rc_0 + rc_p + rc_n
    recall_adv = rc_n / cnt
    foolrate = rc_p / (rc_p + rc_n)
    return recall_adv, foolrate


def parse_fgsm_ret(r_codes):
    r_codes = np.array(r_codes)
    cnt = r_codes.shape[0]
    rc_n = np.sum(r_codes<0)
    rc_0 = np.sum(r_codes==0)
    rc_p = np.sum(r_codes>0)
    recall_adv, foolrate = compute_index(cnt, rc_0, rc_n, rc_p)
    return recall_adv, foolrate


def parse_jsma_ret(r_codes, bit):
    r_codes = np.array(r_codes)
    cnt = r_codes.shape[0]
    rc_n = np.sum(r_codes<0) + np.sum(r_codes>bit)
    rc_0 = np.sum(r_codes==0)
    rc_p = np.sum((r_codes>0) & (r_codes<=bit))
    # print("attack_alg:jsma\tmax_bits:{}".format(bit))
    recall_adv, foolrate = compute_index(cnt, rc_0, rc_n, rc_p)
    return recall_adv, foolrate


def run(out_file):
    with open(out_file, 'r') as f:
        data = json.load(f)
    parse_rets= []
    for report in data:
        alg = report['alg']
        target_model = os.path.split(report['model_file'])[1].split('_')[0]
        if 'retrain' in report['model_file']:
            target_model = 'bpnn_retrain'
        r_codes = report['r_codes']
        max_bit = report['max_bit']

        if alg == 'jsma':
            for modify_bit in range(10, max_bit+10, 10):
                recall, foolrate = parse_jsma_ret(r_codes, modify_bit)
                parse_rets.append((alg, target_model, modify_bit, recall, foolrate))
        elif alg == 'fgsm':
            recall, foolrate = parse_fgsm_ret(r_codes)
            parse_rets.append((alg, target_model, max_bit, recall, foolrate))
        else:
            raise('error')
    return parse_rets


if __name__ == '__main__':
    out_file = os.path.join('/home/prj01/c1/Liliangxun/GraphDroid/baseline','attack_peernet_fgsm.logger.json')
    parse_rets = run(out_file)
    # with open('attack_peernet.report.json', 'w') as f:
    #     json.dump(parse_rets, f, indent=4)
    for a,b,c,d,e in parse_rets:
        print('{}\t{}\t{}\t{:.4f}\t{:.4f}'.format(a,b,c,d,e))


# fgsm	PeerNet	10	0.9859	0.0053
# fgsm	PeerNet	20	0.9462	0.0469
# fgsm	PeerNet	30	0.8208	0.1732
# fgsm	PeerNet	40	0.7706	0.2232