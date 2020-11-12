# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, maml_training, loss_functions, modules
from torchmeta.datasets import CelebA
from torchmeta.utils.data import Task, MetaDataLoader, BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop, Normalize


import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='meta_test',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=401,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--fw_weight', type=float, default=1e2,
               help='Weight for the l2 loss term on the weights of the sine network')
p.add_argument('--train_sparsity_range', type=int, nargs='+', default=[10, 200],
               help='Two integers: lowest number of sparse pixels sampled followed by highest number of sparse'
                    'pixels sampled when training the conditional neural process')

p.add_argument('--epochs_til_ckpt', type=int, default=2,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--dataset', type=str, default='celeba_32x32',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity for the hypo-network module')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--hypo_net_nl', default='sine', help='Nonlinearity for hypo network')

p.add_argument('--conv_encoder', action='store_true', default=False, help='Use convolutional encoder process')
p.add_argument('--partial_conv', default=False, help='Set up partial convolutions')
opt = p.parse_args()


assert opt.dataset == 'celeba_32x32'
if opt.conv_encoder: gmode = 'conv_cnp'
else: gmode = 'cnp'

img_dataset = dataio.CelebA(split='train', downsampled=True)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=(32, 32))
generalization_dataset = dataio.ImageGeneralizationWrapper(coord_dataset,
                                                           train_sparsity_range=opt.train_sparsity_range,
                                                           generalization_mode=gmode)
image_resolution = (32, 32)

dataloader = DataLoader(generalization_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)


num_shots = 100
num_shots_test = 100
batch_size = 30
num_train_per_class = 200
num_test_per_class = 200

transform = Compose([CenterCrop((178, 178)), Resize((32, 32)),
                     ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
dataset_transform = ClassSplitter(shuffle=True,
                                  num_train_per_class=num_shots,
                                  num_test_per_class=num_shots_test)

image_resolution = (32, 32)
dataset = CelebA(num_samples_per_task=1000, transform=transform,
                 target_transform=transform)
split_dataset = ClassSplitter(dataset, num_train_per_class=num_train_per_class,
                              num_test_per_class=num_test_per_class)
meta_dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size)


model = meta_modules.MAMLEncoder(in_features=dataset.img_channels + 2,
                                 out_features=dataset.img_channels,
                                 image_resolution=image_resolution,
                                 hypo_net_nl=opt.hypo_net_nl)


model.cuda()

# Define the loss
loss_fn = partial(loss_functions.image_hypernetwork_loss, None, opt.kl_weight, opt.fw_weight)
summary_fn = partial(utils.write_image_maml, image_resolution, None)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

maml_training.train(model=model, train_dataloader=meta_dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True)
