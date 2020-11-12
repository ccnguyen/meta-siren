import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import os
import diff_operators
from torchvision.utils import make_grid, save_image
import skimage.metrics
import cv2
import meta_modules
import scipy.io.wavfile as wavfile
import cmapy


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)


def densely_sample_activations(model, num_dim=1, num_steps=int(1e6)):
    input = torch.linspace(-1., 1., steps=num_steps).float()

    if num_dim == 1:
        input = input[...,None]
    else:
        input = torch.stack(torch.meshgrid(*(input for _ in num_dim)), dim=-1).view(-1, num_dim)

    input = {'coords':input[None,:].cuda()}
    with torch.no_grad():
        activations = model.forward_with_activations(input)['activations']
    return activations


def write_image_summary_small(image_resolution, mask, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    if mask is None:
        gt_img = dataio.lin2img(gt['img'], image_resolution)
        gt_dense = gt_img
    else:
        gt_img = dataio.lin2img(gt['img'], image_resolution) * mask
        gt_dense = gt_img

    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)

    with torch.no_grad():
        img_gradient = torch.autograd.grad(model_output['model_out'], [model_output['model_in']],
                                           grad_outputs=torch.ones_like(model_output['model_out']), create_graph=True,
                                           retain_graph=True)[0]

        grad_norm = img_gradient.norm(dim=-1, keepdim=True)
        grad_norm = dataio.lin2img(grad_norm, image_resolution)
        writer.add_image(prefix + 'pred_grad_norm', make_grid(grad_norm, scale_each=False, normalize=True),
                         global_step=total_steps)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    write_psnr(pred_img, gt_dense, writer, total_steps, prefix + 'img_dense_')

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    min_max_summary(prefix + 'gt_img', gt_img, writer, total_steps)

    hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix)



def write_image_maml(image_resolution, mask, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    if mask is None:
        gt_img = dataio.lin2img(gt['img'], image_resolution)
        gt_dense = gt_img
    else:
        gt_img = dataio.lin2img(gt['img'], image_resolution) * mask
        gt_dense = gt_img

    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)


    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    write_psnr(pred_img, gt_dense, writer, total_steps, prefix + 'img_dense_')

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    min_max_summary(prefix + 'gt_img', gt_img, writer, total_steps)

    hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix)


def write_func_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train'):
    gt_func = torch.squeeze(gt['func'])
    pred_func = torch.squeeze(model_output['model_out'])
    coords = torch.squeeze(model_output['model_in'].clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3,1)

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, writer, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, writer, total_steps)

    hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix)


def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig



def hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    with torch.no_grad():
        hypo_parameters, embedding = model.get_hypo_net_weights(model_input)

        for name, param in hypo_parameters.items():
            writer.add_histogram(prefix + name, param.cpu(), global_step=total_steps)

        writer.add_histogram(prefix + 'latent_code', embedding.cpu(), global_step=total_steps)



def write_image_summary(image_resolution, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_'):
    gt_img = dataio.lin2img(gt['img'], image_resolution)
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)

    img_gradient = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    img_laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    pred_img = dataio.rescale_img((pred_img+1)/2, mode='clamp').permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    pred_grad = dataio.grads2img(dataio.lin2img(img_gradient)).permute(1,2,0).squeeze().detach().cpu().numpy()
    pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
                             dataio.lin2img(img_laplace), perc=2).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    gt_img = dataio.rescale_img((gt_img+1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    gt_grad = dataio.grads2img(dataio.lin2img(gt['gradients'])).permute(1, 2, 0).squeeze().detach().cpu().numpy()
    gt_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
        dataio.lin2img(gt['laplace']), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_grad', torch.from_numpy(pred_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_lapl).permute(2,0,1), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', torch.from_numpy(gt_img).permute(2,0,1), global_step=total_steps)
    writer.add_image(prefix + 'gt_grad', torch.from_numpy(gt_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_lapl', torch.from_numpy(gt_lapl).permute(2, 0, 1), global_step=total_steps)

    write_psnr(dataio.lin2img(model_output['model_out'], image_resolution),
               dataio.lin2img(gt['img'], image_resolution), writer, total_steps, prefix+'img_')


def write_audio_summary(logging_root_path, model, model_input, gt, model_output, writer, total_steps, prefix='train'):
    gt_func = torch.squeeze(gt['func'])
    gt_rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
    gt_scale = torch.squeeze(gt['scale']).detach().cpu().numpy()
    pred_func = torch.squeeze(model_output['model_out'])
    coords = torch.squeeze(model_output['model_in'].clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3,1)

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, writer, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, writer, total_steps)

    # write audio files:
    wavfile.write(os.path.join(logging_root_path, 'gt.wav'), gt_rate, gt_func.detach().cpu().numpy())
    wavfile.write(os.path.join(logging_root_path, 'pred.wav'), gt_rate, pred_func.detach().cpu().numpy())


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)
