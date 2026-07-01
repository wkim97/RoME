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

## Overview

Multi-perturbation adversarial training (MAT) seeks robustness against multiple adversarial threats
at once, but suffers from **robustness trade-offs** across threats. RoME routes each threat
through a distinct model pathway via a **Mixture of Experts**, using three components:

- **Robust low-rank experts** — experts are low-rank additive updates over a shared backbone, so
  the backbone captures threat-common features while experts learn threat-specific ones.
- **Threat-distinguishing dual-scale gating** — gating fuses local patch-level and global
  image-level features to better tell threats apart.
- **Threat-guided gating diversification** — a regularizer that pushes different threats to
  distinct expert combinations, avoiding threat-agnostic routing.

RoME is modular (plugs into MAT recipes such as MAX and RANDOM), and because threat labels are
used only at training time, it also generalizes to **unseen** threats at inference.

## Key results

- **Union robustness + natural accuracy.** RoME+MAX improves union robustness and natural accuracy over baselines across CIFAR-10, ImageNet-100, and ImageNet-1K datasets.
- **Unseen threats.** Outperforms prior MAT on common corruptions, $\ell_0$, perceptual, patch,
  spatial, and semantic attacks, as well as an adaptive gating-misrouting attack.
- **Architecture-agnostic.** Consistent gains on both Transformer and CNN-based models.

See the paper for full tables and analysis.

## Code

🚧 Code will be released upon publication. Stay tuned.

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

Low-rank experts build on [LoRA](https://github.com/microsoft/LoRA). We thank the authors of the
MAT baselines and evaluation suites whose code and protocols we build upon.
