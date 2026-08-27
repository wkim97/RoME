"""Per-dataset configuration."""

DATASETS = {
    'cifar10': {
        'img_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'mlp_gate': True,
        'eps': {'Linf': 8. / 255., 'L2': 0.5, 'L1': 12.},
    },
    'imagenet100': {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 100,
        'mlp_gate': True,
        'eps': {'Linf': 4. / 255., 'L2': 2.0, 'L1': 255.},
    },
    'imagenet1k': {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'mlp_gate': True,
        'eps': {'Linf': 4. / 255., 'L2': 2.0, 'L1': 255.},
    },
}

NORMS = ('Linf', 'L1', 'L2')


def get_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset {dataset!r}; expected one of {list(DATASETS)}.')
    return DATASETS[dataset]
