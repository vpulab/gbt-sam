### *Requirements*
CUDA version: 11.8.

torch==1.13.0 ; torchaudio==0.13.0 ; torchvision==0.14.0.


### *Data acquisition*
Download any BraTS dataset from Synapse following their instructions.

Link: https://www.synapse.org/Synapse:syn51156910/wiki/627000

The dataset folder structure should be as follows:

```
-data
--brats #Adult Glioma Segmentation
--brats_ssa #Subsaharan Glioma Segmentation
--brats_ped #Pediatric Glioma Segmentation
--brats_men #Meningioma Segmentation
```

You can download as much BraTS datasets as you want to use, but all of them should be placed inside the `data` folder, that should be placed at the same level as the folder of this repo, following the above specified naming instructions.

### *Download code & Set the environment*
Open a terminal and execute the following commands:


```
git clone https://github.com/vpulab/gbt-sam/;
cd gbt-sam;
conda env create -f environment.yml;
conda activate gbt_sam_env;
```

### *Training & Testing*

**Training**

The weights of our final model can be downloaded through [this link](https://drive.google.com/file/d/1ZDmPF8NHaUgZ--xe1a8gYKXc9vEgKRnW/view?usp=sharing).

*** Step 1: Just Patch Embedding

```
python train.py -net sam -mod sam_patch -exp_name gbt-sam_patch-embed -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -b 1 -dataset brats -thd True -num_sample 1 -w 8
```

*** Step 2: Patch Embedding + LoRA + Depth Condition block

```
python train.py -net sam -mod sam_lora_depth -thd True -exp_name  gbt-sam-2step-training -sam_ckpt logs/model_id/Model/best_dice -dataset brats -mid_dim 12 -slice_distance 1 -four_chan True -box True -overlap 75;
```

Parameter `model_id` should be replaced by the saved the model whose weights (from step 1) you want to use. Parameter `mod` can also be defined as: `sam_lora` to train just LoRA blocks and Patch Embedding (no Depth Condition block); or `sam` in case you want to maintain the original SAM architecture. Parameter `four_chan` should be defined as `True` if you want to use all 4 MRI modalities; or `False` if just taking e of them to not train the Patch Embedding Layer. Parameter `dataset` must be defined as any of the names indicated in the 'Data acquisition' section. Parameter `mid_dim` defines the rank of LoRA blocks. 

After running the training command, 'sam_vit_b_01ec64.pth' will be downloaded and stored in 'checkpoint/sam/'. The saved model parameters will be placed in the 'logs/' directory.

**Validation**

```
python val.py -net sam -mod sam_lora_depth -sam_ckpt logs/model_id/Model/best_dice  -weights logs/model_id/Model/best_dice -dataset brats -w 0 -save_individual_global_results experiment_id  -thd True  -mode Validation -mid_dim 12 -slice_distance 1 -four_chan True -box True -overlap 75;

```

Parameter `model_id` should be replaced by the saved model whose weights you want to use. Parameter `save_individual_global_results` is used to create en experiment_id.xlsx containg the results of both per-patient predictions and the mean of the whole dataset. This results will be saved in the 'results_excel/' directory.

