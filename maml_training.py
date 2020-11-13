'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
import time
import numpy as np
import os
import shutil
from torchmeta.utils.gradient_based import gradient_update_parameters
import sys
import imageio


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def gradient_update(model, inner_loss, step_size, params=None):
    if params is None:
        params = OrderedDict(model.named_parameters())

    grads = torch.autograd.grad(inner_loss['img_loss'],
                                params.values(), create_graph=True)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            print(param)
            print(step_size)
            print(grad)
            updated_params[name] = param - step_size * grad
    return updated_params

def adapt(model, loss_fn, model_input, gt, num_adaptation_steps, step_size, writer, summary_fn):
    params = OrderedDict(model.named_parameters())

    results = {'inner_losses': np.zeros(
        (num_adaptation_steps,), dtype=np.float32)}

    for step in range(num_adaptation_steps):
        model_output = model(model_input, test=False, params=params)
        inner_loss = loss_fn(model_output, gt)
        img_loss = inner_loss['img_loss']
        results['inner_losses'][step] = img_loss #inner_loss.item()
        writer.add_scalar('inner_loss', results['inner_losses'][step], step)

        for loss_name, loss in inner_loss.items():
            single_loss = loss.mean()
            writer.add_scalar(f'inner_loop_{loss_name}', single_loss, step)
        summary_fn(model, model_input, gt, model_output, writer, step, inner=True)


        model.zero_grad()
        img_loss.backward()

        updated_params = OrderedDict()
        for name, param in model.named_parameters():
            updated_params[name] = param
        # params = gradient_update(model, inner_loss, step_size, params=params)
        # params = gradient_update_parameters(model, inner_loss,
        #                                     step_size=step_size, params=params)

    return updated_params, results


def normalize(input):
    input = (input-input.min()) / (input.max() - input.min()).float()
    input -= 0.5
    input *= 2.0
    return input


def convert_metadata(batch):
    train_inputs, train_targets = batch

    img_sub = train_targets[:, :, -1, :]
    target = train_targets[:, :, :-1, :]
    target = target[:, 0, :, :]
    # sub = gt[0,:,:].reshape([32,32,3]).permute(0,1,2).detach().numpy()
    # print(sub.shape)
    # imageio.imwrite('test3.png', sub)
    # sys.exit()
    gt = {}
    gt['img'] = target

    model_input = {}
    model_input['coords_sub'] = normalize(train_inputs)
    model_input['img_sub'] = img_sub

    train_sparsity_range = (10, 200)
    ctxt_mask = torch.empty((train_inputs.shape[0], train_sparsity_range[1], 1))
    for i in range(train_inputs.shape[0]):
        subsamples = np.random.randint(train_sparsity_range[0], train_sparsity_range[1])
        rand_idcs = np.random.choice(train_sparsity_range[1], size=subsamples, replace=False)
        ctxt_mask[i, :, :] = torch.zeros(train_sparsity_range[1], 1)
        ctxt_mask[i, rand_idcs, 0] = 1.0
    model_input['ctxt_mask'] = ctxt_mask
    model_input['idx'] = torch.Tensor(np.random.randint(0, 100, size=(train_targets.shape[0])))

    coords = get_mgrid((32, 32))
    coords = coords.unsqueeze(0).repeat(train_targets.shape[0], 1, 1)
    model_input['coords'] = coords

    model_input = {key: value.cuda() for key, value in model_input.items()}
    gt = {key: value.cuda() for key, value in gt.items()}
    return model_input, gt

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False,
          loss_schedules=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        #val = 'y'
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, batch in enumerate(train_dataloader):
                start_time = time.time()

                model_input, gt = convert_metadata(batch['train'])
                test_model_input, test_gt = convert_metadata(batch['test'])

                _, train_targets = batch['train']
                _, test_targets = batch['test']


                num_tasks = test_targets.size(0)
                num_adaptation_steps = 10
                step_size = 0.001
                results = {
                    'num_tasks': num_tasks,
                    'inner_losses': np.zeros((num_adaptation_steps,
                                              num_tasks), dtype=np.float32),
                    'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
                    'mean_outer_loss': 0.
                }
                mean_outer_loss = torch.tensor(0.).cuda()

                for task_num in range(train_targets.shape[0]):
                    params, adaptation_results = adapt(model, loss_fn, model_input, gt,
                                                       num_adaptation_steps= num_adaptation_steps,
                                                       step_size=step_size, writer=writer, summary_fn=summary_fn)
                    results['inner_losses'][:, task_num] = adaptation_results['inner_losses']

                    # do the same processing with the test dataset
                    test_model_output = model(test_model_input, test=True, params=params)
                    outer_loss = loss_fn(test_model_output, test_gt)
                    img_loss = outer_loss['img_loss']
                    results['outer_losses'][task_num] = img_loss
                    mean_outer_loss += img_loss
                mean_outer_loss.div_(task_num)


                results['mean_outer_loss'] = mean_outer_loss.item()
                writer.add_scalar('mean_outer_loss', results['mean_outer_loss'], step)

                mean_outer_loss.backward()
                optim.step()

                ###################
                train_loss = 0.0
                for loss_name, loss in outer_loss.items():

                    single_loss = loss.mean()
                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, test_model_input, test_gt, test_model_output, writer, total_steps, inner=False)

                # if not use_lbfgs:
                #     optim.zero_grad()
                #     train_loss.backward()
                #
                #     optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                    epoch, train_loss, time.time() - start_time))

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
