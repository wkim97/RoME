"""Multi-perturbation adversarial training.

    python train.py --dataset cifar10 --strategy max --data-dir ./datasets
"""

import argparse
import glob
import os
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from attacks import apgd_train
from data import build_loader
from loss import gate_diversity_loss
from models import build_model
from utils.config import NORMS, get_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'imagenet100', 'imagenet1k'])
    p.add_argument('--strategy', default='max', choices=['max', 'random'],
                   help='how the three per-threat attacks are reduced to one batch')
    p.add_argument('--data-dir', default='./datasets')
    p.add_argument('--output-dir', default='./runs')
    p.add_argument('--name', default=None, help='run directory name (default: <strategy>_<dataset>)')

    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-schedule', default='warmup-linear',
                   choices=['warmup-linear', 'constant'])
    p.add_argument('--warmup-epochs', type=int, default=4,
                   help='linear warmup length, in epochs')
    p.add_argument('--final-epochs', type=int, default=4,
                   help='length of the final decay phase, in epochs')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--iters', default='15 15 15',
                   help='APGD iterations for Linf / L1 / L2')

    p.add_argument('--gate', default=None, choices=['mlp', 'linear'],
                   help='gate head type (default: the dataset setting in utils/config.py)')
    p.add_argument('--num-lora', type=int, default=4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--s', type=float, default=4.0)
    p.add_argument('--b', type=float, default=6.0)
    p.add_argument('--lam', type=float, default=0.1, help='gate-diversity loss weight')

    p.add_argument('--num-examples', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def lr_at(args, epoch_f):
    """Learning rate at fractional epoch ``epoch_f`` in ``[0, args.epochs)``."""
    if args.lr_schedule == 'constant':
        return args.lr

    warmup, final = args.warmup_epochs, args.final_epochs
    middle_end = max(args.epochs - final, warmup)

    if epoch_f < warmup:
        return args.lr * epoch_f / max(warmup, 1e-8)
    if epoch_f < middle_end:
        frac = (epoch_f - warmup) / max(middle_end - warmup, 1e-8)
        return args.lr * (1. - frac * (1. - 0.05))
    frac = (epoch_f - middle_end) / max(args.epochs - middle_end, 1e-8)
    return args.lr * (0.05 - frac * (0.05 - 0.001))


def make_adversarial_batch(model, x, y, strategy, epss, iters):
    """Returns ``(x_adv, threat_ids)`` with ``threat_ids`` indexing ``NORMS``."""
    if strategy == 'max':
        advs, losses = [], []
        for norm, eps, n_iter in zip(NORMS, epss, iters):
            x_adv, loss_best, _ = apgd_train(model, x, y, norm=norm, eps=eps,
                                             n_iter=n_iter, is_train=True)
            advs.append(x_adv)
            losses.append(loss_best)
        threat_ids = torch.stack(losses, dim=0).argmax(dim=0).long()
        x_adv = torch.stack(advs, dim=0)[threat_ids, torch.arange(x.shape[0], device=x.device)]
        return x_adv, threat_ids

    threat_ids = torch.randint(0, len(NORMS), (x.shape[0],), device=x.device).long()
    x_adv = torch.zeros_like(x)
    for t, (norm, eps, n_iter) in enumerate(zip(NORMS, epss, iters)):
        mask = threat_ids == t
        if not mask.any():
            continue
        x_adv[mask], _, _ = apgd_train(model, x[mask], y[mask], norm=norm, eps=eps,
                                       n_iter=n_iter, is_train=True)
    return x_adv, threat_ids


def find_latest_checkpoint(run_dir):
    best = (0, None)
    for path in glob.glob(os.path.join(run_dir, 'ep_*.pth')):
        match = re.search(r'ep_(\d+)\.pth$', os.path.basename(path))
        if match and int(match.group(1)) > best[0]:
            best = (int(match.group(1)), path)
    return best


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = get_config(args.dataset)
    epss = [cfg['eps'][n] for n in NORMS]
    iters = [int(v) for v in args.iters.split()]
    if len(iters) != len(NORMS):
        raise ValueError(f'--iters needs {len(NORMS)} values (for {", ".join(NORMS)}).')

    run_dir = os.path.join(args.output_dir, args.name or f'{args.strategy}_{args.dataset}')
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'log_train.txt')

    train_loader = build_loader(args.dataset, args.data_dir, train=True,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                num_examples=args.num_examples, seed=args.seed)

    model = build_model(args.dataset, pretrained=True, r=args.rank,
                        lora_alpha=args.lora_alpha, num_lora=args.num_lora,
                        s=args.s, b=args.b,
                        mlp_gate=None if args.gate is None else args.gate == 'mlp').cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, resume_path = find_latest_checkpoint(run_dir)
    if resume_path:
        ckpt = torch.load(resume_path, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optim'])
        print(f'Resuming from {resume_path} (epoch {start_epoch})')

    print(f'[{args.strategy}] {args.dataset} | ' + ', '.join(
        f'{n} eps={e:.5f} iters={i}' for n, e, i in zip(NORMS, epss, iters)))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.time()
        running_loss = running_gate = running_correct = seen = 0.

        with tqdm(train_loader, unit='batch', desc=f'epoch {epoch}/{args.epochs}') as bar:
            for step, (x, y) in enumerate(bar):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                epoch_f = epoch - 1 + (step + 1) / len(train_loader)
                for group in optimizer.param_groups:
                    group['lr'] = lr_at(args, epoch_f)

                model.eval()
                x_adv, threat_ids = make_adversarial_batch(
                    model, x, y, args.strategy, epss, iters)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, gates = model(x_adv)
                loss = criterion(logits, y)

                gate_loss = gate_diversity_loss(gates, threat_ids, num_threats=len(NORMS))
                if torch.is_tensor(gate_loss):
                    loss = loss + args.lam * gate_loss
                    running_gate += gate_loss.item()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_correct += (logits.argmax(dim=-1) == y).sum().item()
                seen += y.shape[0]
                bar.set_postfix(loss=running_loss / (step + 1),
                                gate=running_gate / (step + 1),
                                acc=running_correct / seen)

        torch.save({'state_dict': model.state_dict(), 'optim': optimizer.state_dict()},
                   os.path.join(run_dir, f'ep_{epoch}.pth'))

        line = (f'[epoch] {epoch} [time] {time.time() - started:.1f}s '
                f'[loss] {running_loss / len(train_loader):.5f} '
                f'[adv acc] {running_correct / seen:.4f}')
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
