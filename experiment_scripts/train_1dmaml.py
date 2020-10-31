# Enable import from parent package

'''
Use set encoder + MAML to regress on higher order polynomial
'''


import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='maml',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=100)
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

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity for the hypo-network module')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

opt = p.parse_args()

dataset = dataio.Polynomial()
coord_dataset = dataio.ImplicitPolyWrapper(dataset)
generalization_dataset = dataio.PolyGeneralizationWrapper(coord_dataset,
                                                            train_sparsity_range=opt.train_sparsity_range)

dataloader = DataLoader(generalization_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = meta_modules.FCEncoder1DHypernet(in_features=1,
                                         out_features=1)

model.cuda()

loss_fn = partial(loss_functions.func_hypernetwork_loss, opt.kl_weight, opt.fw_weight)
summary_fn = partial(utils.write_func_summary)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True)

