# GBT-SAM: A Parameter-Efficient Depth-Aware Model for Generalizable Brain Tumor Segmentation on mp-MRI


## Abstract
[cite_start]GBT-SAM is a parameter-efficient deep learning framework that adapts the large-scale Segment Anything Model (SAM) to volumetric mp-MRI data[cite: 17]. [cite_start]Standard models often fail to fully exploit the multi-parametric MRI (mp-MRI) information and inter-slice contextual data[cite: 15]. [cite_start]Our approach addresses these limitations by leveraging all four MRI modalities (T1, T2, T1c, and T2-FLAIR) and introducing a depth-aware module to capture inter-slice correlations, all while maintaining high parameter efficiency[cite: 18, 19, 72].

## Key Contributions

* [cite_start]**Multi-modal Adaptation for mp-MRI:** We modify the foundational SAM patch embedding layer to accommodate a genuine 4-channel input, enabling the joint processing of all standard mp-MRI sequences without information loss[cite: 72].
* [cite_start]**Depth-Conditioned Correlation Modelling:** We introduce a lightweight Depth-Condition block integrated at multiple architectural stages to efficiently capture inter-slice volumetric dependencies across adjacent slices[cite: 73, 74].
* [cite_start]**Parameter-Efficient Domain Adaptation:** Using Low-Rank Adaptation (LoRA), our model achieves competitive segmentation accuracy with only 9.97M trainable parameters, the lowest among existing SAM-based approaches[cite: 19, 76].
* [cite_start]**Robust Domain Generalization:** Our framework is validated across four distinct clinical domains (Adult Glioma, Meningioma, Pediatric Glioma, and Sub-Saharan Glioma), demonstrating superior domain robustness and zero-shot transfer capabilities[cite: 21, 77].

## Architecture Overview
[cite_start]GBT-SAM employs a two-step fine-tuning strategy[cite: 19]. [cite_start]In the first step, the architecture is frozen except for the modified patch embedding layer, which is optimized to process the full mp-MRI input[cite: 262]. [cite_start]In the second step, we integrate LoRA blocks within the encoder layers and fine-tune the patch embedding alongside these lightweight adapters and the Depth-Condition module[cite: 264]. 

[cite_start]During training, we utilize a slice selection strategy that processes only 4 slices per scan to reduce computational complexity while retaining essential tumor-related information[cite: 18]. [cite_start]At inference, the model utilizes the full volume via a sliding window approach to ensure clinically robust 3D segmentation[cite: 181, 277].

## Performance
[cite_start]GBT-SAM trained exclusively on the BraTS Adult Glioma dataset achieves a Dice score of 92.66[cite: 20]. [cite_start]It demonstrates exceptional efficiency, requiring significantly fewer trainable parameters than top-performing alternatives while maintaining state-of-the-art segmentation accuracy[cite: 76].

## Cite
If you use this code or our approach in your research, please cite our paper:

```bibtex
@inproceedings{cdiana2024med-sam-brain,
  title={How SAM Perceives Different mp-MRI Brain Tumor Domains?},
  author={Diana-Albelda, Cecilia and Alcover-Couso, Roberto and García-Martín, Álvaro and Bescos, Jesus},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4959--4970},
  year={2024}
}
