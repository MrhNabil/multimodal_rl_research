# Multimodal-RL-VQA

> **Compositional Skill Learning in Multimodal Reinforcement Learning for Visual Question Answering**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-green.svg)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This repository investigates whether **reinforcement learning can compose frozen vision skills with trainable language skills** for Visual Question Answering (VQA), inspired by the paper *"From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones"*.

### Key Research Question

> Can RL compose pretrained visual features (CLIP) with trainable classification heads to answer visual questions without intermediate supervision?

### Key Results

| Method | Accuracy | Details |
|--------|----------|---------|
| **Supervised Learning** | **74.0%** | Cross-entropy loss, 1000 steps |
| RL (REINFORCE) | 53.7% | Policy gradient, 3000 steps |
| Frozen Baseline | 0.2% | No training |

**Finding**: Supervised learning outperforms RL by 20+ percentage points on this multimodal task, suggesting cross-modal skill composition is more challenging than within-text composition.

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐
│   Image     │───▶│  CLIP ViT    │───▶│ Projection │───▶│  Fusion     │───▶ Answer
│  224×224    │    │  (FROZEN)    │    │    MLP     │    │   Layer     │
└─────────────┘    │   151M       │    │ (trainable)│    └─────────────┘
                   └──────────────┘    └────────────┘          ▲
                                                               │
                   ┌─────────────────────────────────────────┘
                   │
              ┌────────────┐
              │  Question  │
              │   Type     │
              └────────────┘
```

- **Vision Encoder**: CLIP ViT-B/32 (frozen, 151M parameters)
- **Projection Layer**: MLP (trainable, ~500K parameters)
- **Classification Heads**: 4 type-specific heads (color, shape, count, spatial)

---

## 📊 Experiments Summary

I conducted **61+ experiments** across:

### Training Methods
- Frozen baseline (no training)
- Supervised learning (cross-entropy)
- Reinforcement learning (REINFORCE)

### Learning Rate Sensitivity
| Learning Rate | RL Accuracy |
|---------------|------------|
| 1e-5 | 29.4% |
| 1e-4 | 45.2% |
| **2e-4** | **53.7%** (optimal) |
| 1e-3 | 29.3% |
| 1e-2 | 14.2% |

### Per-Question-Type Performance
| Type | Supervised | RL |
|------|------------|-----|
| Count | 82.0% | 58.0% |
| Shape | 77.4% | 71.8% |
| Color | 75.7% | **20.6%** |
| Spatial | 61.3% | 39.8% |

**Key Finding**: RL struggles severely with color questions (20.6%) but performs well on shape (71.8%).

---

## 📁 Repository Structure

```
multimodal-rl-vqa/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── src/                         # Source code
│   ├── models/                  # Model architectures
│   │   ├── vqa_model.py         # Main VQA model
│   │   ├── clip_encoder.py      # CLIP wrapper
│   │   └── classification_heads.py
│   ├── training/                # Training logic
│   │   ├── supervised.py        # Supervised training
│   │   ├── reinforce.py         # REINFORCE algorithm
│   │   └── trainer.py           # Main trainer class
│   ├── data/                    # Data handling
│   │   ├── dataset.py           # VQA dataset class
│   │   └── preprocessing.py     # Data preprocessing
│   └── utils/                   # Utilities
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Plotting functions
│
├── configs/                     # Experiment configurations
│   ├── supervised.yaml          # Supervised training config
│   └── reinforce.yaml           # RL training config
│
├── scripts/                     # Runnable scripts
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── prepare_data.py          # Data preparation
│
├── experiments/                 # Experiment results
│   ├── results/                 # Raw results (JSON)
│   └── figures/                 # Generated plots
│
├── docs/                        # Documentation
│   ├── IEEE_PROJECT_REPORT.pdf  # Full IEEE paper
│   └── IEEE_PROJECT_REPORT.tex  # LaTeX source
│
└── notebooks/                   # Jupyter notebooks
    └── analysis.ipynb           # Results analysis
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/multimodal-rl-vqa.git
cd multimodal-rl-vqa
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py --num_samples 5000
```

### 3. Train Models

**Supervised Learning:**
```bash
python scripts/train.py --method supervised --lr 0.0002 --steps 1000
```

**Reinforcement Learning:**
```bash
python scripts/train.py --method reinforce --lr 0.0002 --steps 3000
```

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pt
```

---

## 📈 Results Visualization

### Method Comparison
![Method Comparison](experiments/figures/method_comparison.png)

### Learning Rate Sensitivity
![LR Sensitivity](experiments/figures/learning_rate.png)

### Per-Type Accuracy
![Per-Type](experiments/figures/per_type.png)

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{nabil2024multimodal,
  author = {Rakib Hossain Nabil},
  title = {Compositional Skill Learning in Multimodal Reinforcement Learning for Visual Question Answering},
  year = {2024},
  institution = {North South University},
  url = {https://github.com/yourusername/multimodal-rl-vqa}
}
```

---

## 🔑 Key Findings

1. **Supervised > RL**: 74.0% vs 53.7% accuracy
2. **Optimal LR for RL**: 2e-4 (too high causes collapse)
3. **Skill-specific composability**: Shape (72%) composes better than color (21%) via RL
4. **More data doesn't help**: 50K samples gave 61.5% (worse than 5K with 68.7%)
5. **Frozen features limit spatial reasoning**: 24-61% on spatial questions

---

## 📚 References

1. Anonymous. "From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL." arXiv:2509.25123, 2024.
2. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021.
3. Williams. "Simple Statistical Gradient-Following Algorithms for RL." Machine Learning, 1992.
4. Antol et al. "VQA: Visual Question Answering." ICCV, 2015.
5. Johnson et al. "CLEVR: A Diagnostic Dataset for Compositional Reasoning." CVPR, 2017.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rakib Hossain Nabil**  
Department of Electrical and Computer Engineering  
North South University, Dhaka, Bangladesh  
ID: 2131005642

---

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- The authors of the compositional learning paper for inspiration
- North South University for support
