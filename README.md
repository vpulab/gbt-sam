# GBT-SAM: A Parameter-Efficient Depth-Aware Model for Generalizable Brain Tumor Segmentation on mp-MRI

This repository contains the official implementation of our paper: [GBT-SAM: A Parameter-Efficient Depth-Aware Model for Generalizable Brain Tumor Segmentation on mp-MRI](https://arxiv.org/abs/2503.04325).

## Abstract
GBT-SAM is a parameter-efficient framework that adapts the large-scale Segment Anything Model (SAM) to volumetric mp-MRI data. We address the limitations of standard models in utilizing multi-parametric MRI (mp-MRI) information and inter-slice contextual data. Our approach integrates all four MRI modalities (T1, T2, T1c, and T2-FLAIR) and introduces a depth-aware module to capture inter-slice correlations, all while maintaining high parameter efficiency.

## Key Contributions
* **High-Performance Efficiency:** We achieve a **92.66% Dice score** using only **9.97M trainable parameters**, making it significantly more efficient than existing SAM-based medical segmentation methods.
* **Multi-modal Adaptation:** We modify the foundational SAM patch embedding layer to accommodate a 4-channel input, enabling the joint processing of all standard mp-MRI sequences without information loss.
* **Depth-Conditioned Correlation Modelling:** We introduce a lightweight Depth-Condition block integrated at multiple architectural stages to efficiently capture inter-slice volumetric dependencies across adjacent slices.
* **Robust Domain Generalization:** Our framework is validated across four distinct clinical domains (Adult Glioma, Meningioma, Pediatric Glioma, and Sub-Saharan Glioma), demonstrating superior domain robustness and zero-shot transfer capabilities.

## Architecture Overview
GBT-SAM employs a two-step fine-tuning strategy:
1. **Patch Embedding Optimization:** The architecture is frozen except for the modified 4-channel patch embedding layer, which is optimized to process the full mp-MRI input. 
2. **LoRA & Depth-Condition Integration:** We integrate LoRA blocks within the encoder layers and fine-tune them alongside the depth-aware modules.

During training, we utilize a slice selection strategy that processes 4 slices per scan to reduce computational complexity. At inference, the model utilizes the full volume via a sliding window approach to ensure clinically robust 3D segmentation.

## Cite
If you use this code or our approach in your research, please cite our work:

```bibtex
@article{diana2025gbtsam,
  title={GBT-SAM: A Parameter-Efficient Depth-Aware Model for Generalizable Brain Tumor Segmentation on mp-MRI},
  author={Diana-Albelda, Cecilia and Alcover-Couso, Roberto and García-Martín, Álvaro and Bescos, Jesus and Escudero-Viñolo, Marcos},
  journal={arXiv preprint arXiv:2503.04325},
  year={2025}
}
