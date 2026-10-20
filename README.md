## 🔥【AAAI 2026】Text-based Aerial-Ground Person Retrieval

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.08369-b31b1b.svg)](https://arxiv.org/abs/2511.08369) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <a href="#-news">News</a> •
  <a href="#-setup">Setup</a> •
  <a href="#-dataset-preparation">Dataset</a> •
  <a href="#-configuration">Config</a> •
  <a href="#-training">Training</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📢 News
* **[2025-12]** Code and dataset released
* **[2025-11]** Paper uploaded to arXiv

---

## Reproduction records for the revised paper (this fork)

This fork reproduces and analyses TAG-CLIP on a single RTX 5090. Complete records of the revision-stage experiments live in [`records/`](records/README.md): per-run `config_used.yaml`, `run_command.txt`, `git_commit.txt`, `environment_summary.txt`, `metrics_by_epoch.csv` and `final_results.txt` for every TAG-PEDES and external-dataset run, the analysis archives behind the tables of the paper and the supplementary material, SHA-256 checksums of all saved checkpoints (`records/checksums.txt`), the ViT-L/14 configuration used by the minimal reproduction command (`records/configs/cfg_l14.yaml`) and the exact Python environment (`records/environment/requirements_tagpr5090.txt`: Python 3.10.12, PyTorch 2.7.1+cu128). `results_ext/` holds the original seed-0 external analyses of 2026-09-01; `records/` holds the complete revision-stage records. The upstream `requirements.txt` below is kept unchanged for reference.
## 🛠️ Setup

### Requirements

*   **Hardware**: 4 × NVIDIA RTX 3090 GPUs (24GB VRAM recommended)
*   **Software**: Python 3.8+ (Recommended), CUDA 11.x+

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Flame-Chasers/TAG-PR.git
cd TAG-PR
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

We provide the processed dataset via Quark Cloud Drive.

1.  **Download**: [Click here to download](https://pan.quark.cn/s/dcddfa17cb7f?pwd=8pE6) (Access Code: `8pE6`)
2.  **Organize**: Extract and arrange the files as follows:

```text
dataset/
├── anno_dir/
│   ├── train_reid.json
│   └── test_reid.json
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

---

## ⚙️ Configuration

Modify the configuration file located at `config/s.config.yaml`.

```yaml
# Data Paths
anno_dir: "/path/to/dataset/anno_dir"      # ⚠️ Absolute path to annotation JSONs
image_dir: "/path/to/dataset/images"       # ⚠️ Absolute path to image directory

# Model Settings
model:
  checkpoint: "/path/to/clip/ViT-B-16.pt"  # Path to pre-trained CLIP weights
  # ... other model params
```

---

## 🚀 Training

We support multi-GPU training and shell script execution.

### Option 1: Torchrun 

Use `torchrun` for distributed data parallel (DDP) training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  --nproc_per_node=4 \
  main.py
```

### Option 2: Shell Script

You can also use the provided shell script wrapper:

```bash
bash shell/train.sh
```

---

## 📝 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{zhou2025text,
  title={Text-based Aerial-Ground Person Retrieval},
  author={Zhou, Xinyu and Wu, Yu and Ma, Jiayao and Wang, Wenhao and Cao, Min and Ye, Mang},
  journal={arXiv preprint arXiv:2511.08369},
  year={2025}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).