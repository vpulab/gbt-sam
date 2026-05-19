#!/usr/bin/env python3
""" 
Evaluate network using PyTorch
Author: Cecilia Diana-Albelda
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg_valid
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
from utils import *
import function


print("=> [DEBUG 1] Imports completed. Reading arguments...", flush=True)
args = cfg_valid.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)

print("=> [DEBUG 2] Arguments read. Initializing network on GPU...", flush=True)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

print("=> [DEBUG 3] Network initialized. Loading weights...", flush=True)

# Load pretrained model weights
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch']

# Handle state_dict keys for distributed vs non-distributed testing
state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Add `module.` prefix to match expected distributed keys
        name = 'module.' + k
        new_state_dict[name] = v
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

# Set up logging directories
args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


# Segmentation data transformations
val_transforms = Compose(
        [
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )
    

if 'brats' in args.dataset: # BraTS dataset 
    # Brain Tumor data configuration
    print(f"=> Initializing Brats Dataset class...", flush=True)
    brats_dataset = Brats(args, args.data_path, mode='Validation', transform=val_transforms)
    
    dataset_size = len(brats_dataset)
    indices = list(range(dataset_size))
    
    # Validation split logic based on the domain
    # For the domain seen during training (Adult Glioma):
    if args.dataset == 'brats':
        np.random.seed(666)
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        test_sampler = SubsetRandomSampler(indices[:split])
    else:
        # For domains unseen during training (Zero-shot generalization):
        test_sampler = SubsetRandomSampler(indices[:])
    
    print('Length test sampler: ', len(test_sampler), flush=True)
    
    print(f"=> Creating DataLoader (workers={args.w}, pin_memory=True)...", flush=True)
    nice_test_loader = DataLoader(brats_dataset, batch_size=args.b, sampler=test_sampler, num_workers=args.w, pin_memory=True)


# Begin evaluation phase
if args.mod: 
    net.eval()
    print("=> Entering function.validation_sam...", flush=True)
    tol, metrics = function.validation_sam(args, nice_test_loader, start_epoch, net)
    
    # Assuming metrics returns (iou, dice). Adjust if it returns more values.
    eiou, edice = metrics[0], metrics[1] 
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')
