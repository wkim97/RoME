"""Gating diversification loss."""

import torch

__all__ = ['gate_diversity_loss']


def _pairwise_neg_distance(per_threat):
    if len(per_threat) < 2:
        return None
    flat = torch.stack(per_threat, dim=0).reshape(len(per_threat), -1)
    off_diag = ~torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
    return -torch.cdist(flat, flat, p=2)[off_diag].mean()


def _threat_means(scores, threat_ids, num_threats):
    means = []
    for t in range(num_threats):
        mask = threat_ids == t
        if mask.any():
            means.append(scores[mask].mean(0))
    return means


def gate_diversity_loss(gates, threat_ids, num_threats=3):
    """Args:
        gates: per-block ``[qkv_aux, proj_aux]`` list returned by the model.
        threat_ids: ``(B,)`` threat index in ``[0, num_threats)`` per sample.

    Returns a scalar tensor, or ``0.0`` when fewer than two threats are present.
    """
    total = 0.0
    n_adapters = 0
    for block in gates:
        for aux in block:
            n_adapters += 1
            terms = [
                _pairwise_neg_distance(
                    _threat_means(aux[key], threat_ids, num_threats))
                for key in ('global_gate_scores', 'local_gate_scores')
            ]
            terms = [t for t in terms if t is not None]
            if terms:
                total = total + sum(terms) / len(terms)
    if n_adapters == 0 or not torch.is_tensor(total):
        return 0.0
    return total / n_adapters
