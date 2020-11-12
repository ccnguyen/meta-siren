# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions
import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./maml_logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=50)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=401,
               help='Number of epochs to train for.')

# MAML specific params
p.add_argument('--num_ways', type=int, default=5, help='Number of classes per task (N in "N-way")')
p.add_argument('--num_shots', type=int, default=5, help='Number of training examples per class (k in "k-shot")')
p.add_argument('--num_shots_test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

opt = p.parse_args()
