# GBT-SAM Setup and Usage Instructions

This document provides the standard operating procedures to configure the environment, acquire the required datasets, and execute the training and validation pipelines for the GBT-SAM framework.

## 1. Requirements

The system must meet the following hardware and software specifications:
* **CUDA Version:** 11.8
* **PyTorch Ecosystem:**
  * `torch == 1.13.0`
  * `torchaudio == 0.13.0`
  * `torchvision == 0.14.0`

## 2. Data Acquisition

The framework is configured to process multi-parametric MRI (mp-MRI) datasets comprising four modalities: T1, T1c, T2, and T2-FLAIR.

1. Download the BraTS datasets from the official Synapse portal: [BraTS 2023 Challenge](https://www.synapse.org/Synapse:syn51156910/wiki/627000).
2. Create a `data` directory at the root level of the cloned repository.
3. Organize the downloaded datasets within the `data` directory using the exact nomenclature specified below:

```text
data/
  ├── brats/       # Adult Glioma Segmentation
  ├── brats_ssa/   # Subsaharan Glioma Segmentation
  ├── brats_ped/   # Pediatric Glioma Segmentation
  └── brats_men/   # Meningioma Segmentation
```

*Note: Ensure the directory names match the expected structure strictly to prevent path resolution errors during execution.*

## 3. Environment Configuration

Execute the following commands in the terminal to clone the repository, generate the Conda environment, and activate it:

```bash
git clone https://github.com/vpulab/gbt-sam/
cd gbt-sam
conda env create -f environment.yml
conda activate gbt_sam_env
```

## 4. Pre-trained Models

For direct inference or to bypass the training phases, the final pre-trained model weights are available for download:
**[Download GBT-SAM Pre-trained Weights](https://drive.google.com/file/d/1ZDmPF8NHaUgZ--xe1a8gYKXc9vEgKRnW/view?usp=sharing)**

## 5. Training Pipeline

GBT-SAM implements a two-step fine-tuning protocol to adapt the baseline SAM architecture to the 4-channel medical domain. 

### Step 1: Patch Embedding Optimization
This phase optimizes exclusively the modified patch embedding layer to accommodate the volumetric 4-channel input.

```bash
python train.py -net sam -mod sam_patch -exp_name gbt-sam_patch-embed -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -b 1 -dataset brats -thd True -num_sample 1 -w 8
```
*Note: During the initial execution, the foundational SAM weights (`sam_vit_b_01ec64.pth`) will be downloaded automatically and stored in the `checkpoint/sam/` directory. Output model parameters are saved in the `logs/` directory.*

### Step 2: Patch Embedding + LoRA + Depth Condition Block
The second phase integrates the Depth-Condition module and LoRA blocks, conducting joint fine-tuning alongside the patch embedding layer.

```bash
python train.py -net sam -mod sam_lora_depth -thd True -exp_name gbt-sam-2step-training -sam_ckpt logs/model_id/Model/best_dice -dataset brats -mid_dim 12 -slice_distance 1 -four_chan True -box True -overlap 75;
```

### Training Parameters Guide:
* `-sam_ckpt`: Path to the baseline checkpoint. Replace `model_id` with the specific directory name generated during Step 1.
* `-mod`: Specifies the architectural configuration.
  * `sam_lora_depth`: Trains Patch Embedding, LoRA blocks, and the Depth Condition block (Default).
  * `sam_lora`: Trains Patch Embedding and LoRA blocks only.
  * `sam_patch`: Trains Patch Embedding only.
  * `sam`: Executes the original unmodified SAM architecture.
* `-dataset`: Indicates the target dataset directory (e.g., `brats`, `brats_ssa`, `brats_ped`, `brats_men`).
* `-four_chan`: Set to `True` to process all 4 mp-MRI modalities simultaneously. Set to `False` to use a single modality.
* `-mid_dim`: Defines the rank for the LoRA blocks (Default: 12).
* `-overlap`: Sets the percentage of overlap between the simulated bounding-box prompt and the ground-truth tumor (Default: 75).
* `-slice_distance`: Defines the interval distance between consecutive slices extracted for depth conditioning (Default: 1).

## 6. Validation and Inference

The inference pipeline processes the complete MRI volumes systematically utilizing a sliding window approach. 

Execute the following command for validation. Ensure `model_id` is replaced with the directory containing the fully trained model from Step 2, and `experiment_id` is defined to identify the output records:

```bash
python val.py -net sam -mod sam_lora_depth -sam_ckpt logs/model_id/Model/best_dice -weights logs/model_id/Model/best_dice -dataset brats -w 0 -save_individual_global_results experiment_id -thd True -mode Validation -mid_dim 12 -slice_distance 1 -four_chan True -box True -overlap 75;
```

### Validation Parameters Guide:
* `-save_individual_global_results`: Compiles the results into an Excel spreadsheet (`experiment_id.xlsx`). This file contains detailed per-patient prediction scores and the aggregated dataset mean. Output files are stored in the `results_excel/` directory.
