# UniRefiner: Teaching Pre-trained ViTs to Self-Dispose Dross via Contrastive Register

[[Project Page](https://congpeiqiu.github.io/UniRefiner/)] [[Paper](https://arxiv.org/abs/2605.19622)] [[PDF](https://arxiv.org/pdf/2605.19622)] [[BibTeX](#citation)]

## Introduction

UniRefiner is a one-for-all refinement framework for ViT foundation models across architectures and scales. It improves dense spatial representations by teaching pre-trained ViTs to redirect spurious tokens into contrastive registers. Please see the [project page](https://congpeiqiu.github.io/UniRefiner/) for the full method description and visual analysis.

| EVA-CLIP-8B | SigLIP2-So400M | RICE-ViT |
|:---:|:---:|:---:|
| <img src="assets/readme/eva_clip_8b_refine.webp" width="260"> | <img src="assets/readme/siglip2_so400m_refine.webp" width="260"> | <img src="assets/readme/rice_vit_refine.webp" width="260"> |

## Installation

We recommend using `uv` through the provided setup helper:

```bash
bash tools/setup_uv_env.sh
source .venv/bin/activate
```

Or install manually:

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
uv pip install -e ".[dev]"
```

## Data Preparation

UniRefiner training only needs an image folder. Images are loaded recursively; annotations are not required.

```text
UniRefiner/
├── data/
│   └── train_images/
│       ├── image_000001.jpg
│       ├── image_000002.png
│       └── subfolder/
│           └── image_000003.webp
└── assets/
    └── backgrounds/
        └── fixed_reference.png
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. 

## Run

Running UniRefiner with 4 GPUs for the SigLIP2-So400M backbone:

```bash
PYTHONPATH=$PWD \
torchrun --nproc_per_node=4 -m unirefiner.cli.train \
  --config configs/siglip2_so400m.yaml \
  --override data.train_image_root=/path/to/train_images \
  --override experiment.output_dir=outputs/siglip2_so400m \
  --override logging.wandb=true \
  --override logging.wandb_mode=online \
  --override logging.wandb_project=UniRefiner \
  --override logging.wandb_run_name=siglip2_so400m \
  --override diagnostics.vis_pca_interval=100
```

Backbones are loaded through Hugging Face `transformers` by default. The project supports ViT-style foundation models when a wrapper exposes the minimal UniRefiner interface:

- `encode_dense(images)`: returns dense raster-order patch tokens with shape `[B, N, C]`.
- `patch_size`: patch size used to map image resolution to token grids.
- `image_mean` and `image_std`: preprocessing statistics.
- `hook_prepare(...)`: optional; only needed when attention hijacker-hijackee filtering is enabled.

Built-in wrappers cover the recipes listed below. Custom wrappers can be passed through `model.wrapper` as `module.path:object`.



## Training

| # | Backbone | Recipe | Checkpoint |
|:---:|:---|:---|:---:|
| 1 | EVA-CLIP-8B | [configs/evaclip8b.yaml](configs/evaclip8b.yaml) | TBA |
| 2 | InternViT | [configs/internvit_6b_224px.yaml](configs/internvit_6b_224px.yaml) | TBA |
| 3 | OpenAI / LAION CLIP | [configs/laion_clip_giant.yaml](configs/laion_clip_giant.yaml) | TBA |
| 4 | DINOv2-Giant | [configs/dinov2_giant.yaml](configs/dinov2_giant.yaml) | TBA |
| 5 | SigLIP2-So400M | [configs/siglip2_so400m.yaml](configs/siglip2_so400m.yaml) | TBA |
| 6 | SigLIP2-Giant | [configs/siglip2_giant_384.yaml](configs/siglip2_giant_384.yaml) | TBA |
| 7 | RICE-ViT | [configs/rice_vit_large_560.yaml](configs/rice_vit_large_560.yaml) | TBA |


## License

This project is licensed under the [Apache License 2.0](LICENSE). 

## Citation

```bibtex
@article{qiu2026unirefiner,
  title={UniRefiner: Teaching Pre-trained ViTs to Self-Dispose Dross via Contrastive Register},
  author={Qiu, Congpei and Hu, Zhaoyu and Ke, Wei and Tian, Zhuotao and Wu, Yanhao and Zhang, Tong},
  journal={arXiv preprint arXiv:2605.19622},
  year={2026},
  eprint={2605.19622},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.19622}
}
```
