""" 
Author: Cecilia Diana-Albelda
"""

import argparse
import pandas as pd
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
import models.sam.utils.transforms as samtrans
from conf import settings
from utils import *
from loss import EDiceLoss

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True


def train_sam(args, net: nn.Module, optimizer,  train_loader,
          epoch, writer, vis = 50):
    hard = 0
    epoch_loss_values = 0.0
    epoch_loss = 0.0
    ind = 0
    
    # Train mode
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # Set up loss function depending on the thd flag
    if args.thd:
        sigmoid = nn.Sigmoid()
        lossfunc = nn.BCELoss() 
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            feat_name = pack['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            
            # If not enough GPU, uncomment the following 3 lines
            num_slices = 4
            i_slices =  SliceSelection(num_slices, masks, args.slice_distance)
            # i_slices = SelectSlices(num_slices, masks)
            imgs = imgs[:,:,:,:,i_slices] 
            masks = masks[:,:,:,:,i_slices]
            

            # Handle point prompt generation if not provided in the pack
            if 'pt' not in pack:
                a = masks
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            # Process tensor dimensions for volumetric settings
            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                point_labels = torch.ones(imgs.size(0))
                # Project generated points to the new image size
                pt = torch.Tensor(numpy.array([((pt[i].detach().cpu().numpy()*(args.out_size,args.out_size))/masks.shape[2:]) for i in range (pt.shape[0])]))
                imgs = torchvision.transforms.Resize((args.image_size,args.image_size), antialias=None)(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size), antialias=None)(masks)
            
            # Handle bounding box processing
            if args.box == 'True' and 'box' not in pack:
                boxes = CalculateBoxes(masks, args.overlap)
                transform = samtrans.ResizeLongestSide(target_length=args.out_size)
                boxes = torch.as_tensor(transform.apply_boxes_torch(boxes, (imgs.shape[-2],imgs.shape[-1])), dtype=torch.float, device=GPUdevice)
                # SHOW IMAGE WITH BOX TO VERIFY (Debugging)
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            # Format point coordinates and labels into proper tensor shapes for the prompt encoder
            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice) # shape: (b_size, 2) 
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice) # shape: (b_size) 
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch) # shape: (1, b_size, 2) 

            # Initialization
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            
            # Gradients setup based on the training mode strategy
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if ("Adapter" in n): 
                        value.requires_grad = True
                    else:
                        value.requires_grad = False
            if args.mod == 'sam':
                for n, value in net.image_encoder.named_parameters():
                    value.requires_grad = False 
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora' or args.mod == 'sam_lora_depth':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder) 
                
                if args.mod == 'sam_lora_depth':
                    for n, value in net.image_encoder.named_parameters():
                        if 'depth_adapter' in n:
                            value.requires_grad = True

                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True

            if args.four_chan == True:
                for n, value in net.image_encoder.named_parameters(): 
                    if ('patch_embed' in n): # 1st layer 
                        value.requires_grad = True 
            
            # pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            # print('Trainable params: ', round(pytorch_total_params/1000000,2))
            # exit()

            imge= net.image_encoder(imgs)
            
            with torch.no_grad():
                if args.net == "sam":
                    if args.box == 'True':
                        se, de = net.prompt_encoder(
                            points=None,
                            boxes=boxes,
                            masks=None,
                        )
                    else:
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
                
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
 
            # Resize to the ordered output size
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))

            loss = lossfunc(sigmoid(pred), masks)

            # Option to save individual predictions
            if args.save_preds == 1:
                content = [np.sum(masks.squeeze()[i,:,:]) for i in range (masks.squeeze().shape[0])] 
                i_slice = np.argmax(content)
                mask_slice = masks.squeeze()[i_slice,:,:]
                pred_slice = pred.squeeze()[i_slice,:,:]
                t2f_slice = imgs[3, i_slice,:,:] #3 = t2f 
                loss_slice = lossfunc(sigmoid(pred_slice), mask_slice)

                if args.box == 'True':
                    box_slice = boxes[i_slice,:] 
                    view_preds(mask_slice.detach().cpu(), pred_slice.detach().cpu(), t2f_slice.detach().cpu(), feat_name, [0,0], box_slice.detach().cpu(), loss_slice.detach().cpu().numpy())
                else:
                    view_preds(mask_slice.detach().cpu(), pred_slice.detach().cpu(), t2f_slice.detach().cpu(), feat_name, pt_slice.detach().cpu().numpy(), [0,0,0,0], loss_slice.detach().cpu().numpy())
            
            epoch_loss += loss.item()
            epoch_loss_values += 1
            pbar.set_postfix(**{'loss (batch)': loss})

            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()

            current_lr = args.lr 
            
            optimizer.zero_grad()

            # Visualize images for tensorboard or sanity checks
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return epoch_loss/epoch_loss_values, current_lr

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    net.eval()
    mask_type = torch.float32
    n_val = len(val_loader)
    
    # Initialization adapted to val.py unpacking (assuming eval_seg returns 2 values: iou, dice)
    # If eval_seg returns 4, val.py must be adjusted. Initialized to 0 depending on the required length.
    ave_res, mix_res = (0,0,0,0), (0,0,0,0) 
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    
    if args.thd:
        sigmoid = nn.Sigmoid()
        lossfunc = nn.BCELoss() 
    else:
        lossfunc = criterion_G

    patient_results_list = []
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            feat_name = name[0].split('/')[-1]

            print(f"\n[Patient {ind+1}/{n_val}] Loaded: {feat_name}. Starting 3D inference...", flush=True)

            # Strict management of volumetric prompts
            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            num_slices = imgsw.size(-1)
            batch_size = 4
            
            full_pred = torch.zeros((imgsw.size(0), 1, imgsw.size(2), imgsw.size(3), num_slices), dtype=torch.float32, device=GPUdevice)

            for z in range(0, num_slices, batch_size):
                chunk_imgs = imgsw[..., z : z + batch_size]
                chunk_masks = masksw[..., z : z + batch_size]
                
                if args.thd:
                    chunk_pts = ptw[:, :, z : z + batch_size]
                else:
                    chunk_pts = ptw
                
                actual_chunk_size = chunk_imgs.size(-1)
                
                # Image, mask and point tensor padding
                if actual_chunk_size < batch_size:
                    pad_size = batch_size - actual_chunk_size
                    chunk_imgs = F.pad(chunk_imgs, (0, pad_size), "constant", 0)
                    chunk_masks = F.pad(chunk_masks, (0, pad_size), "constant", 0)
                    if args.thd:
                        chunk_pts = F.pad(chunk_pts, (0, pad_size), "constant", -1) # -1 to invalidate ghost points

                if args.thd:
                    imgs = rearrange(chunk_imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(chunk_masks, 'b c h w d -> (b d) c h w ')
                    pt = rearrange(chunk_pts, 'b n d -> (b d) n')
                    
                    point_labels_batch = torch.ones(imgs.size(0), device=GPUdevice)
                    pt = torch.Tensor(np.array([((pt[i].detach().cpu().numpy()*(args.out_size,args.out_size))/masks.shape[2:]) for i in range (pt.shape[0])])).to(GPUdevice)
                    
                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size), antialias=None)(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size), antialias=None)(masks)

                if args.box == 'True':
                    boxes = CalculateBoxes(masks, args.overlap)
                    transform = samtrans.ResizeLongestSide(target_length=masks.shape[-1])
                    boxes = torch.as_tensor(transform.apply_boxes_torch(boxes, (imgs.shape[-2], imgs.shape[-1])), dtype=torch.float, device=GPUdevice)

                imgs = imgs.to(dtype=mask_type, device=GPUdevice)
                
                # Point formatting for the network
                if args.box != 'True' and point_labels_batch[0] != -1:
                    coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels_batch, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt_formatted = (coords_torch, labels_torch)
                else:
                    pt_formatted = None

                with torch.no_grad():
                    imge = net.image_encoder(imgs)
                    
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        if args.box == 'True':
                            se, de = net.prompt_encoder(points=None, boxes=boxes, masks=None)
                        else:
                            se, de = net.prompt_encoder(points=pt_formatted, boxes=None, masks=None)

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        )
                
                    pred = F.interpolate(pred, size=(imgsw.shape[2], imgsw.shape[3]))
                    pred_3d = rearrange(pred, '(b d) c h w -> b c h w d', b=imgsw.size(0), d=batch_size)
                    
                    full_pred[..., z : z + actual_chunk_size] = pred_3d[..., :actual_chunk_size]

            flat_full_pred = rearrange(full_pred, 'b c h w d -> (b d) c h w')
            flat_masksw = rearrange(masksw, 'b c h w d -> (b d) c h w')

            loss = lossfunc(sigmoid(flat_full_pred), flat_masksw)
            tot += loss.item()

            if args.vis and ind % args.vis == 0:
                namecat = 'Test'
                for na in name:
                    img_name = na.split('/')[-1].split('.')[0]
                    namecat = namecat + img_name + '+'
                
                mid_z = num_slices // 2
                # Strict 2D extraction for visualization
                img_mid = imgsw[..., mid_z]
                pred_mid = full_pred[..., mid_z]
                mask_mid = masksw[..., mid_z]
                
                vis_image(img_mid, pred_mid, mask_mid, 
                          os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)

            temp = eval_seg(flat_full_pred, flat_masksw, threshold)
            
            patient_dice = temp[1].item() if torch.is_tensor(temp[1]) else temp[1]
            patient_results_list.append({
                'Patient_ID': feat_name,
                'DICE': patient_dice
            })
            
            # Compatibility check with val.py
            if len(mix_res) != len(temp):
                mix_res = tuple([0] * len(temp))
                
            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    pbar.update()

    # ==========================================
    # NEW: SAVING RESULTS IN EXCEL
    # ==========================================
    if hasattr(args, 'save_individual_global_results') and args.save_individual_global_results != '':
        exp_id = args.save_individual_global_results
        
        # Dynamic path: creates 'excel_results' in the same folder as function.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'excel_results')
        os.makedirs(results_dir, exist_ok=True)

        # 1. Individual Excel
        df_individual = pd.DataFrame(patient_results_list)
        indiv_path = os.path.join(results_dir, f"{exp_id}.xlsx")
        df_individual.to_excel(indiv_path, index=False)
        print(f"\n=> Individual results saved at: {indiv_path}", flush=True)

        # 2. Global Excel
        global_path = os.path.join(results_dir, "0_all_ablation_results.xlsx")
        global_dice = df_individual['DICE'].mean()

        new_row = pd.DataFrame({
            'Experiment_ID': [exp_id],
            'Global_DICE': [global_dice]
        })

        if os.path.exists(global_path):
            df_global = pd.read_excel(global_path)
            if exp_id in df_global['Experiment_ID'].values:
                df_global.loc[df_global['Experiment_ID'] == exp_id, 'Global_DICE'] = global_dice
            else:
                df_global = pd.concat([df_global, new_row], ignore_index=True)
        else:
            df_global = new_row

        df_global.to_excel(global_path, index=False)
        print(f"=> Global mean ({global_dice:.4f}) recorded at: {global_path}\n", flush=True)
    # ==========================================

    return tot / n_val ,  tuple([a/n_val for a in mix_res])


def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )
