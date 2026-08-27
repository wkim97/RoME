# RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations

Official implementation of our **ECCV 2026** paper:

> **RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations**
>
> Woo Jae Kim¹, Kyle Min², Suhyeon Ha¹, Joonsung Jeon¹, Sung-eui Yoon¹
>
> ¹KAIST, ²Oracle

<p align="left">
  <a href="https://github.com/wkim97/RoME"><img src="https://img.shields.io/badge/code-RoME-black?logo=github" alt="code"></a>
  <img src="https://img.shields.io/badge/venue-ECCV%202026-blue" alt="ECCV 2026">
</p>

---

## Setup

```bash
conda create -n rome python=3.11 -y
conda activate rome
pip install -r requirements.txt
```

## Data

```
datasets/
├── cifar-10-batches-py/          # auto-downloaded
├── imagenet_100/{train,val}/     # one folder per class
└── imagenet/{train,val}/         # one folder per class
```

Override the root with `DATA_DIR=/path/to/datasets` or `--data-dir`.

## Checkpoints

Download from [Google Drive](https://drive.google.com/drive/folders/1OnuDO-y3Bf3_F_SQKEHVW7zU9icniPEN?usp=sharing)
and unpack into `checkpoints/`:

```
checkpoints/
├── ViT_CIFAR10/{MAX,RANDOM}_apgd_cifar10.pth
├── ViT_ImageNet100/{MAX,RANDOM}_apgd_imagenet100.pth
└── ViT_ImageNet1K/{MAX,RANDOM}_apgd_imagenet1k.pth
```

Override the root with `CKPT_DIR=/path/to/checkpoints` or `--ckpt`.

## Training

```bash
./scripts/train_cifar10.sh     max        # or: random
./scripts/train_imagenet100.sh max
./scripts/train_imagenet1k.sh  max
```

Second argument is the GPU index. Each script holds the hyper-parameters for its dataset;
per-threat budgets live in `utils/config.py`. Checkpoints and `log_train.txt` go to
`runs/<strategy>_<dataset>/`, and an interrupted run resumes from the latest epoch.

To override anything, call `train.py` directly:

```bash
python train.py --dataset cifar10 --strategy max --data-dir ./datasets
python train.py --help
```

## Evaluation

AutoAttack (standard) under ℓ∞, ℓ2 and ℓ1; reports clean, per-threat, average and union accuracy.

```bash
./scripts/eval_final.sh cifar10_max
./scripts/eval_final.sh imagenet100_random
./scripts/eval_final.sh imagenet1k_max 1        # second argument = GPU index
```

Results are written to `logs/<mode>.txt`.

```bash
python evaluate.py --dataset cifar10 \
    --ckpt checkpoints/ViT_CIFAR10/MAX_apgd_cifar10.pth --data-dir ./datasets
python evaluate.py --help
```

Expert count, LoRA rank and gate type are read from the checkpoint. `--s` / `--b` set
β(l) = σ(s(1 − l/L) − b) and must match the values used for training.

## Layout

```
train.py                     adversarial training
evaluate.py                  AutoAttack evaluation
attacks/apgd.py              APGD
models/vit.py                ViT-B/16 backbone
models/lora_moe_mlp.py       LoRA-MoE adapter, MLP gates
models/lora_moe_linear.py    LoRA-MoE adapter, linear gates
loss/gate_diversity.py       gating diversification loss
data/loaders.py              dataset loaders
utils/config.py              per-dataset geometry and ε
scripts/                     entry points
```

## Citation

```bibtex
@inproceedings{kim2026rome,
  title     = {RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations},
  author    = {Kim, Woo Jae and Min, Kyle and Ha, Suhyeon and Jeon, Joonsung and Yoon, Sung-eui},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

Built on [LoRA](https://github.com/microsoft/LoRA), [timm](https://github.com/huggingface/pytorch-image-models)
and [AutoAttack](https://github.com/fra31/auto-attack).
