"""AutoAttack evaluation under the L_inf / L_2 / L_1 union.

    python evaluate.py --dataset cifar10 --ckpt checkpoints/ViT_CIFAR10/MAX_apgd_cifar10.pth

``--s`` and ``--b`` are not stored in the checkpoint and must match training.
"""

import argparse
import os

import torch
import torch.nn as nn
from autoattack import AutoAttack
from tqdm import tqdm

from data import build_loader
from models import load_checkpoint
from utils.config import get_config

EVAL_ORDER = ('Linf', 'L2', 'L1')


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'imagenet100', 'imagenet1k'])
    p.add_argument('--data-dir', default='./datasets')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-examples', type=int, default=0,
                   help='seeded random subset of the test split; 0 uses all of it')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--s', type=float, default=4.0)
    p.add_argument('--b', type=float, default=6.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--log-path', default=None)
    return p.parse_args()


@torch.no_grad()
def clean_accuracy(model, loader, device):
    correct = []
    for x, y in tqdm(loader, desc='Clean'):
        x, y = x.to(device), y.to(device)
        correct.append(model(x).argmax(dim=1) == y)
    return torch.cat(correct)


def autoattack_accuracy(model, loader, norm, eps, device):
    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', device=device)
    correct = []
    for x, y in tqdm(loader, desc=f'AutoAttack {norm} eps={eps:.5f}'):
        x, y = x.to(device), y.to(device)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
        with torch.no_grad():
            correct.append(model(x_adv).argmax(dim=1) == y)
    return torch.cat(correct)


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    log_path = args.log_path
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def report(text):
        print(text)
        if log_path:
            with open(log_path, 'a') as f:
                f.write(text + '\n')

    loader = build_loader(args.dataset, args.data_dir, train=False,
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          num_examples=args.num_examples, seed=args.seed)

    model = load_checkpoint(args.ckpt, args.dataset, s=args.s, b=args.b).to(device)
    model.eval()
    model = LogitsOnly(model).to(device).eval()

    epss = get_config(args.dataset)['eps']
    report(f'== {args.ckpt} | {args.dataset} | {len(loader.dataset)} images | '
           f's={args.s} b={args.b} ==')

    correct = clean_accuracy(model, loader, device)
    report(f'Clean Acc: {correct.float().mean():.4f}')

    union = torch.ones_like(correct)
    accs = []
    for norm in EVAL_ORDER:
        correct = autoattack_accuracy(model, loader, norm, epss[norm], device)
        accs.append(correct.float().mean().item())
        union &= correct
        report(f'AutoAttack {norm} (eps={epss[norm]:.5f}) Acc: {accs[-1]:.4f}')

    report(f'AutoAttack Average Acc ({", ".join(EVAL_ORDER)}): {sum(accs) / len(accs):.4f}')
    report(f'AutoAttack Union Acc ({" and ".join(EVAL_ORDER)}): {union.float().mean():.4f}')


if __name__ == '__main__':
    main()
