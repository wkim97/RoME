"""Model construction and checkpoint loading."""

import torch

from utils.config import get_config
from .lora_moe_linear import LoRAMoEAdapterLinear
from .lora_moe_mlp import LoRAMoEAdapterMLP
from .vit import VisionTransformer, vit_base_patch16_224

__all__ = ['LoRAMoEAdapterMLP', 'LoRAMoEAdapterLinear', 'VisionTransformer',
           'build_model', 'inspect_checkpoint', 'load_checkpoint']


def build_model(dataset, pretrained=True, r=16, lora_alpha=16, num_lora=4,
                s=4.0, b=6.0, mlp_gate=None, gate_proj_dim=100):
    cfg = get_config(dataset)
    return vit_base_patch16_224(
        pretrained=pretrained,
        img_size=cfg['img_size'],
        patch_size=cfg['patch_size'],
        num_classes=cfg['num_classes'],
        mlp_gate=cfg['mlp_gate'] if mlp_gate is None else mlp_gate,
        r=r, lora_alpha=lora_alpha, num_lora=num_lora, s=s, b=b,
        gate_proj_dim=gate_proj_dim)


def _read_state_dict(path, map_location='cpu'):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        obj = obj['state_dict']
    if not isinstance(obj, dict):
        raise TypeError(f'{path} does not hold a state dict (got {type(obj).__name__}).')
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in obj.items()}


def inspect_checkpoint(state_dict):
    """Read the adapter hyper-parameters back off the parameter shapes."""
    prefix = next(k.rsplit('.lora_As.', 1)[0] for k in state_dict if '.lora_As.' in k)
    return {
        'num_lora': sum(1 for k in state_dict if k.startswith(f'{prefix}.lora_As.')),
        'r': state_dict[f'{prefix}.lora_As.0.weight'].shape[0],
        'mlp_gate': f'{prefix}.global_gate.0.weight' in state_dict,
        'num_classes': state_dict['head.weight'].shape[0],
    }


def load_checkpoint(path, dataset, s=4.0, b=6.0, lora_alpha=None, pretrained=False,
                    verbose=True):
    """Build the model a checkpoint expects, load it, and return it.

    ``lora_alpha`` is not recoverable from the parameter shapes and defaults to ``r``.
    """
    state_dict = _read_state_dict(path)
    spec = inspect_checkpoint(state_dict)

    expected = get_config(dataset)['num_classes']
    if spec['num_classes'] != expected:
        raise ValueError(
            f'{path} has a {spec["num_classes"]}-way head but --dataset {dataset} '
            f'expects {expected}.')

    model = build_model(dataset, pretrained=pretrained, r=spec['r'],
                        lora_alpha=lora_alpha if lora_alpha is not None else spec['r'],
                        num_lora=spec['num_lora'], s=s, b=b,
                        mlp_gate=spec['mlp_gate'])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        gate = 'mlp' if spec['mlp_gate'] else 'linear'
        print(f'Loaded {path}\n  num_lora={spec["num_lora"]} r={spec["r"]} '
              f'gate={gate} s={s} b={b}')
        missing = [k for k in missing if 'global_proj' not in k]
        if missing:
            print(f'  missing keys ({len(missing)}): {missing[:5]}')
        if unexpected:
            print(f'  unexpected keys ({len(unexpected)}): {unexpected[:5]}')
    return model
