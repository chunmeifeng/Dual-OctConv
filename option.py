import os
import argparse

parser = argparse.ArgumentParser()


# Model specifications
parser.add_argument('--n_GPU', type=int, default=2,
                    help='Number of GPUs')
parser.add_argument('--name', '-n', type=str, default='DeepComplex_B10_lrb4_cpd320_1Un3_grappa',
                    help='Model name')
parser.add_argument('--alpha', '-a', type=float, default=0.125,
                    help='alpha of OctConv, proportion of low frequency')
parser.add_argument('--n_blocks', type=int, default=1,
                    help='number of OctComplex blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')                    
# Data specifications
parser.add_argument('--data_dir', type=str, default='/newdisk/chunmeifeng/yzy/datasets/MRI',  # ./brain/AXT2/256
                    help='dataset directory')
parser.add_argument('--data_dst', default='coronal_pd_320',
                    help='coronal_pd, knee, brain, ...')
parser.add_argument('--contrast', '-c', type=str, default=None,
                    help="'AXFLAIR', 'AXT1', 'AXT1POST', 'AXT1PRE', 'AXT2'")      
parser.add_argument('--data_size', type=int, default=320,
                    help='input data size or shape')
parser.add_argument('--in_channels', '-ic', type=int, default=32,
                    help='input data channels')
parser.add_argument('--augment', action='store_true',
                    help='do not use data augmentation')                    
parser.add_argument('--mask_name', type=str, default='new1Ca3_256',
                    help='Unsampling mask name')

# Training specifications
parser.add_argument('--epochs', type=int, default=50,
                help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--eval_every', type=int, default=140,
                    help='do eval per N batches')
parser.add_argument('--test_every', type=int, default=1,
                    help='do test per N batches')
parser.add_argument('--everyslices', action='store_true',
                    help='save results based on every slices')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
                    
# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                help='Number of epochs to train')
parser.add_argument('--lr_decay_rate', type=float, default=0.95,
                help='learning decay rate')                    

# Loss specifications
parser.add_argument('--loss', type=str, default='L1',
                help='option: L1, L2, etc...')

# Option for finetuning
parser.add_argument('--test_only', '-te_o',action='store_true',
                    help='set this option to test the model')
parser.add_argument('--rsname', '-rsn', default='DeepComplex_B10_lrb4_cpd320_1Un3_grappa',
                    help='Continue training from a specified epoch')
parser.add_argument('--reset_lr', type=float, default=None,
                help='Continue training but change the based learning rate')
parser.add_argument('--reset_op', type=str, default=None,
                help='Continue training but change the optimizer')


args = parser.parse_args()

def check_model_name():
    """
    检查args.name中是否带有这些名称，若有，则修改为目标数据集
    """
    if args.name.find('coronal_pd')>=0 or args.name.find('crpd')>=0:
        print('Dataset was changed from {} to coronal_pd(crppd)'.format(args.data_dst))
        args.data_dst = 'coronal_pd'
    elif args.name.find('knee')>=0:
        print('Dataset was changed from {} to fastMRI knee'.format(args.data_dst))
        args.data_dst = 'knee'
    elif args.name.find('brain')>=0:
        print('Dataset was changed from {} to fastMRI brain'.format(args.data_dst))
        args.data_dst = 'brain'

def template_set():
    if args.data_dst == 'coronal_pd_320' or args.data_dst == 'coronal_pd_320_c0' or args.data_dst == 'coronal_pd_320aug' or args.data_dst == 'sagittal_t2_320_s1':
        print('######## Now Datasets is old version of knee in 320 ########')
        args.data_dir = os.path.join(args.data_dir, args.data_dst)
        args.data_size = 320
        args.in_channels = 30
        args.nb_samples = 340
        args.nb_train = 280
        args.nb_val = args.nb_samples - args.nb_train
    elif args.data_dst == 'coronal_pd' or args.data_dst == 'coronal_pd_fs' or args.data_dst == 'axial_t2' or args.data_dst == 'sagittal_pd' or args.data_dst == 'sagittal_t2':
        print('######## Now Datasets is old version of knee in 256 ########')
        args.data_dir = os.path.join(args.data_dir, args.data_dst)
        args.data_size = 256
        args.in_channels = 30
        args.nb_samples = 340 
        args.nb_train = 280
        args.nb_val = args.nb_samples - args.nb_train

def check_mask_name():
    if args.mask_name in ['1Ca2', '1Ca3', '1Ca4', '1Un2.98', '2Rd005', '2Rd1', '2Rd2', '2Rd3', '2Rd4',
                          'Pr1', 'Pr2', 'Pr3', 'Pr4', 'Pr5', 'new1Ca3_320', 'new1Ca3_256',
                          '1Un2_320', '1Un3_320', '2Rd2_320', '2Rd3_320', '2Rd4_320', '2Rd5_320', '1Ca2_320', '1Ca3_320', '1Ca4_320',
                          '2Dradial4', '2Dradial6', '2Dradial9',
                          ]:
        args.mask_name = {'1Ca2': '1D-Cartesian-0.2',
                          '1Ca3': '1D-Cartesian-0.3',
                          '1Ca4': '1D-Cartesian-0.4',
                          '1Un2.98': '1Duniform2.98_ac29',
                          '2Rd005': '2D-random-0.05',
                          '2Rd1': '2D-random-0.1',
                          '2Rd2': '2D-random-0.2',
                          '2Rd3': '2D-random-0.3',
                          '2Rd4': '2D-random-0.4',
                          'Pr1': 'Pseudo_radial_0.1',
                          'Pr2': 'Pseudo_radial_0.2',
                          'Pr3': 'Pseudo_radial_0.3',
                          'Pr4': 'Pseudo_radial_0.4',
                          'Pr5': 'Pseudo_radial_0.5',
                          'new1Ca3_256': 'new1D-Cartesian-0.3_256',
                          'new1Ca3_320': 'new1D-Cartesian-0.3_320',
                          '1Un2_320': '1D-Uniform-0.2_320',
                          '1Un3_320': '1D-Uniform-0.3_320',
                          '2Rd2_320': '2D-Random-0.2_320',
                          '2Rd3_320': '2D-Random-0.3_320',
                          '2Rd4_320': '2D-Random-0.4_320',
                          '2Rd5_320': '2D-Random-0.5_320',
                          '1Ca2_320': '1D-Cartesian-0.2_320',
                          '1Ca3_320': '1D-Cartesian-0.3_320',
                          '1Ca4_320': '1D-Cartesian-0.4_320',
                          '2Dradial4': 'radial_mask_4x', 
                          '2Dradial6': 'radial_mask_6x', 
                          '2Dradial9': 'radial_mask_9x'}.get(args.mask_name)

args.mask_path = os.path.join(args.data_dir, 'mask')

check_model_name()
template_set()
check_mask_name()

if args.epochs == 0:
    args.epochs = 50

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

# args.gpus = args.gpus.split('.')