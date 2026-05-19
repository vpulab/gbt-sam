""" 
Configuration file with arguments for validation
Author: Cecilia Diana-Albelda
"""

import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    
    # Network Architecture Configurations
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-encoder', type=str, default='default', help='encoder type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', default='msa_test_isic', type=str, help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=0, help='visualization')
    
    # Adversarial / Pretraining Flags
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    
    # Hardware and GPU Configurations
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-distributed', default='none', type=str, help='multi GPU ids to use')
    
    # Training and Validation Hyperparameters
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-val_freq', type=int, default=5, help='interval between each validation')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    
    # DataLoader arguments tailored for validation
    parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
    
    # Dimension and Feature Extraction Configurations
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    
    # Checkpoints and Weight Loading
    parser.add_argument('-weights', type=str, default=0, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default=0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default=0, help='the weights sim')
    parser.add_argument('-sam_ckpt', default=None, help='sam checkpoint address')
    
    # Medical Imaging and Dataset Specifics
    parser.add_argument('-dataset', default='brats_ssa', type=str, help='dataset name')
    parser.add_argument('-data_path', type=str, default='../data', help='The path of segmentation data')
    parser.add_argument('-thd', type=bool, default=False, help='3d or not')
    parser.add_argument('-chunk', type=int, default=100, help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4, help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96, help='resolution of roi')
    parser.add_argument('-evl_chunk', type=int, default=None, help='evaluation chunk')
    parser.add_argument('-four_chan', type=bool, default=False, help='training patch embedding as inputs are 4 channel ims')
    parser.add_argument('-mri', type=str, default='', help='which im. is repeated x3 in MRI {t1,t1c,t2,t2f}')
    
    # Specific validation defaults and SAM Architecture Adjustments
    parser.add_argument('-mode', type=str, default='Validation', help='Training or Validation')
    parser.add_argument('-box', type=str, default='False', help='Including bounding box prompt or not')
    parser.add_argument('-tumor_region', type=str, default='wt', help='region of the tumor to segment {et,tc,wt}')
    parser.add_argument('-slice_distance', type=int, default=-1, help='Distance between slices chosen')
    parser.add_argument('-overlap', type=int, default=100, help='percentage of overlap of the BB with the tumor')
    parser.add_argument('-mid_dim', type=int, default=None, help='middle dim of adapter or the rank of lora matrix')
    
    # Execution and Logging Flags
    parser.add_argument('-save_preds', type=int, default=0, help='Save Predictions')
    parser.add_argument('-csv_results', type=int, default=0, help='Create a CSV with loss per patient')
    parser.add_argument('-save_individual_global_results', type=str, default='', help='Experiment ID to save XLSX')

    opt = parser.parse_args()

    return opt
