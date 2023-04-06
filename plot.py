import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from time import time
import os


def artificial_data_reconstruction_plot(model, latent, data, target):
    """
    This function plots 8 figures of a reconstructed latent space, each for a different orientation of the 
    reconstructed latent space.
    """
    model.eval()
    model.cpu()
    z_reconstructed = (model(data)[0]).detach()
    sig = torch.stack([z_reconstructed[target == i].std(
        0, unbiased=False) for i in range(model.n_classes)])
    rms_sig = np.sqrt(np.mean(sig.numpy()**2, 0))
    latent_sig = torch.stack([latent[target == i].std(
        0, unbiased=False) for i in range(model.n_classes)])
    latent_rms_sig = np.sqrt(np.mean(latent_sig.numpy()**2, 0))

    for dim_order in range(2):
        for dim1_factor in [1, -1]:
            for dim2_factor in [1, -1]:
                fig = plt.figure(figsize=(12, 3.5))

                plt.subplot(1, 4, 1)
                plt.scatter(latent[:, 0], latent[:, 1],
                            c=target, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('GROUND TRUTH', fontsize=16, family='serif')

                plt.subplot(1, 4, 2)
                plt.scatter(data[:, 0], data[:, 1], c=target, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('OBSERVED DATA\n(PROJECTION)',
                          fontsize=16, family='serif')

                plt.subplot(1, 4, 3)
                dim1 = np.flip(np.argsort(rms_sig))[dim_order]
                dim2 = np.flip(np.argsort(rms_sig))[(1+dim_order) % 2]
                plt.scatter(
                    dim1_factor*z_reconstructed[:, dim1], dim2_factor*z_reconstructed[:, dim2], c=target, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('RECONSTRUCTION', fontsize=16, family='serif')

                plt.subplot(1, 4, 4)
                plt.semilogy(np.flip(np.sort(rms_sig)), '-ok')
                ground_truth = np.flip(np.sort(latent_rms_sig))
                plt.semilogy(scale_ground_truth(
                    ground_truth, rms_sig), '-ok', alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('SPECTRUM', fontsize=16, family='serif')

                plt.tight_layout()
                fig_idx = 4*dim_order + 2 * \
                    max(dim1_factor, 0) + max(dim2_factor, 0)
                plt.savefig(os.path.join(model.save_dir, 'figures',
                            f'reconstruction_{fig_idx:d}.png'))
                plt.close()


def da_synthetic_data_reconstruction_plot(model, latent, data, target, epoch):
    """
    This function plots 8 figures of a reconstructed latent space, each for a different orientation of the 
    reconstructed latent space.
    """
    model.eval()
    model.cpu()
    z_reconstructed = (model(data)[0]).detach()
    sig = torch.stack([z_reconstructed[target == i].std(
        0, unbiased=False) for i in range(model.n_domains)])
    rms_sig = np.sqrt(np.mean(sig.numpy()**2, 0))
    latent_sig = torch.stack([latent[target == i].std(
        0, unbiased=False) for i in range(model.n_domains)])
    latent_rms_sig = np.sqrt(np.mean(latent_sig.numpy()**2, 0))
    # in da_synthetic experiments, we set dim_c=2 and dim_s=2 in default, which is consistent with the experiment in 'Partial Identiﬁability for Domain Adaptation'
    # in this synthetic experiment, we could tell the estimated C_1 and C_2, so we'll plot all the permutation.
    # first plot with 4*4 subplots, similar to Figure 3 in 'Partial Identiﬁability for Domain Adaptation'
    # fig = plt.figure(figsize=(12, 3.5))
    fig, axs = plt.subplots(4, 4)
    dim_c_1 = np.flip(np.argsort(rms_sig))[0]
    estimated_c_1 = z_reconstructed[:, dim_c_1].reshape([-1, 1])
    dim_c_2 = np.flip(np.argsort(rms_sig))[1]
    estimated_c_2 = z_reconstructed[:, dim_c_2].reshape([-1, 1])
    dim_s_1 = np.flip(np.argsort(rms_sig))[2]
    estimated_s_1 = z_reconstructed[:, dim_s_1].reshape([-1, 1])
    dim_s_2 = np.flip(np.argsort(rms_sig))[3]
    estimated_s_2 = z_reconstructed[:, dim_s_2].reshape([-1, 1])
    estimated_z = torch.cat((estimated_s_1, estimated_s_2, estimated_c_1, estimated_c_2), 1)
    torch.save(estimated_z, os.path.join(model.save_dir, 'data', 'estimated_z_{}.pt'.format(epoch)))
    
    true_s_1 = latent[:, 0:1]
    true_s_2 = latent[:, 1:2]
    true_c_1 = latent[:, 2:3]
    true_c_2 = latent[:, 3:4]
    # dim_s_length = model.dim_s
    
    # top first row
    # x-axis is True C1, y-axis is Estimated C1
    axs[0, 0].scatter(true_c_1, estimated_c_1,
                c=target, s=6, alpha=0.3)
    # axs[0, 0].set_xlabel('True C1')
    axs[0, 0].set_ylabel('Estimated C1')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    # x-axis is True C2, y-axis is Estimated C1
    axs[0, 1].scatter(true_c_2, estimated_c_1,
                c=target, s=6, alpha=0.3)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    # x-axis is True S1, y-axis is Estimated C1
    axs[0, 2].scatter(true_s_1, estimated_c_1,
                c=target, s=6, alpha=0.3)
    axs[0, 2].set_xticks([])
    # axs[0, 2].set_ylabel('True S1')
    axs[0, 2].set_yticks([])
    # x-axis is True S2, y-axis is Estimated C1
    axs[0, 3].scatter(true_s_2, estimated_c_1,
                c=target, s=6, alpha=0.3)
    axs[0, 3].set_xticks([])
    axs[0, 3].set_yticks([])
    
    # top second row
    # x-axis is True C1, y-axis is Estimated C2
    axs[1, 0].scatter(true_c_1, estimated_c_2,
                c=target, s=6, alpha=0.3)
    axs[1, 0].set_ylabel('Estimated C2')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    # x-axis is True C2, y-axis is Estimated C2
    axs[1, 1].scatter(true_c_2, estimated_c_2,
                c=target, s=6, alpha=0.3)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    # x-axis is True S1, y-axis is Estimated C2
    axs[1, 2].scatter(true_s_1, estimated_c_2,
                c=target, s=6, alpha=0.3)
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    # x-axis is True S2, y-axis is Estimated C2
    axs[1, 3].scatter(true_s_2, estimated_c_2,
                c=target, s=6, alpha=0.3)
    axs[1, 3].set_xticks([])
    axs[1, 3].set_yticks([])
    
    # top third row
    # x-axis is True C1, y-axis is Estimated S1
    axs[2, 0].scatter(true_c_1, estimated_s_1,
                c=target, s=6, alpha=0.3)
    axs[2, 0].set_ylabel('Estimated S1')
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    # x-axis is True C2, y-axis is Estimated S1
    axs[2, 1].scatter(true_c_2, estimated_s_1,
                c=target, s=6, alpha=0.3)
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])
    # x-axis is True S1, y-axis is Estimated S1
    axs[2, 2].scatter(true_s_1, estimated_s_1,
                c=target, s=6, alpha=0.3)
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])
    # x-axis is True S2, y-axis is Estimated S1
    axs[2, 3].scatter(true_s_2, estimated_s_1,
                c=target, s=6, alpha=0.3)
    axs[2, 3].set_xticks([])
    axs[2, 3].set_yticks([])
    
    # bottom row
    # x-axis is True C1, y-axis is Estimated S2
    axs[3, 0].scatter(true_c_1, estimated_s_2,
                c=target, s=6, alpha=0.3)
    axs[3, 0].set_ylabel('Estimated S2')
    axs[3, 0].set_xlabel('True C1')
    axs[3, 0].set_xticks([])
    axs[3, 0].set_yticks([])
    # x-axis is True C2, y-axis is Estimated S2
    axs[3, 1].scatter(true_c_2, estimated_s_2,
                c=target, s=6, alpha=0.3)
    axs[3, 1].set_xlabel('True C2')
    axs[3, 1].set_xticks([])
    axs[3, 1].set_yticks([])
    # x-axis is True S1, y-axis is Estimated S2
    axs[3, 2].scatter(true_s_1, estimated_s_2,
                c=target, s=6, alpha=0.3)
    axs[3, 2].set_xlabel('True S1')
    axs[3, 2].set_xticks([])
    axs[3, 2].set_yticks([])
    # x-axis is True S2, y-axis is Estimated S2
    axs[3, 3].scatter(true_s_2, estimated_s_2,
                c=target, s=6, alpha=0.3)
    axs[3, 3].set_xlabel('True S2')
    axs[3, 3].set_xticks([])
    axs[3, 3].set_yticks([])
    
    plt.close()
    fig.savefig(os.path.join(model.save_dir, 'figures', 'reconstruction_{}_0.png'.format(epoch)))
    
    
    # second plot, we'll plot true plot.
    fig = plt.figure(figsize=(12, 3.5))

    plt.subplot(1, 4, 1)
    plt.scatter(true_s_1, true_c_1,
                c=target, s=6, alpha=0.3)
    
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_1')
    plt.xlabel('S_1')
    # plt.title('C1-S1', fontsize=16, family='serif')

    plt.subplot(1, 4, 2)
    # plt.scatter(data[:, 0], data[:, 1], c=target, s=6, alpha=0.3)
    plt.scatter(true_s_2, true_c_1, c=target, s=6, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_1')
    plt.xlabel('S_2')
    # plt.title('C1-S2', fontsize=16, family='serif')

    plt.subplot(1, 4, 3)
    plt.scatter(true_s_1, true_c_2, c=target, s=6, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_2')
    plt.xlabel('S_1')

    plt.subplot(1, 4, 4)
    plt.scatter(true_s_2, true_c_2, c=target, s=6, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_2')
    plt.xlabel('S_2')

    plt.tight_layout()
    # fig_idx = 4*dim_order + 2 * \
    #     max(dim1_factor, 0) + max(dim2_factor, 0)
    plt.savefig(os.path.join(model.save_dir, 'figures', 'reconstruction_{}_1.png'.format(epoch)))
    plt.close()
    
    # third figure
    fig = plt.figure(figsize=(12, 3.5))

    plt.subplot(1, 4, 1)
    plt.scatter(estimated_s_1, estimated_c_1,
                c=target, s=6, alpha=0.3)
    
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_1')
    plt.xlabel('S_1')
    # plt.title('C1-S1', fontsize=16, family='serif')

    plt.subplot(1, 4, 2)
    # plt.scatter(data[:, 0], data[:, 1], c=target, s=6, alpha=0.3)
    plt.scatter(estimated_s_2, estimated_c_1, c=target, s=6, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_1')
    plt.xlabel('S_2')
    # plt.title('C1-S2', fontsize=16, family='serif')

    plt.subplot(1, 4, 3)
    plt.scatter(estimated_s_1, estimated_c_2, c=target, s=6, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_2')
    plt.xlabel('S_1')

    plt.subplot(1, 4, 4)
    plt.scatter(estimated_s_2, estimated_c_2, c=target, s=6, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('C_2')
    plt.xlabel('S_2')

    plt.tight_layout()
    # fig_idx = 4*dim_order + 2 * \
    #     max(dim1_factor, 0) + max(dim2_factor, 0)
    plt.savefig(os.path.join(model.save_dir, 'figures', 'reconstruction_{}_2.png'.format(epoch)))
    plt.close()
    
    # fourth figure
    # plt.subplot(1, 4, 4)
    plt.semilogy(np.flip(np.sort(rms_sig)), '-ok')
    ground_truth = np.flip(np.sort(latent_rms_sig))
    ground_truth = latent_rms_sig
    plt.semilogy(scale_ground_truth(
        ground_truth, rms_sig), '-ok', alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.title('SPECTRUM', fontsize=16, family='serif')
    plt.savefig(os.path.join(model.save_dir, 'figures', 'reconstruction_{}_3.png'.format(epoch)))
    plt.close()
    
    
    # for dim_order in range(2):
    #     for dim1_factor in [1, -1]:
    #         for dim2_factor in [1, -1]:
    #             fig = plt.figure(figsize=(12, 3.5))

    #             plt.subplot(1, 4, 1)
    #             plt.scatter(latent[:, 0], latent[:, model.dim_s],
    #                         c=target, s=6, alpha=0.3)
                
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.title('GROUND TRUTH', fontsize=16, family='serif')

    #             plt.subplot(1, 4, 2)
    #             # plt.scatter(data[:, 0], data[:, 1], c=target, s=6, alpha=0.3)
    #             plt.scatter(data[:, 0], data[:, model.dim_s], c=target, s=6, alpha=0.3)
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.title('OBSERVED DATA\n(PROJECTION)',
    #                       fontsize=16, family='serif')

    #             plt.subplot(1, 4, 3)
    #             dim1 = np.flip(np.argsort(rms_sig))[dim_order]
    #             dim2 = np.flip(np.argsort(rms_sig))[(1+dim_order) % 2]
    #             dim_c_start = np.flip(np.argsort(rms_sig))[0]
    #             dim_s_start = np.flip(np.argsort(rms_sig))[model.dim_c]
    #             plt.scatter(
    #                 dim1_factor*z_reconstructed[:, dim_s_start], dim2_factor*z_reconstructed[:, dim_c_start], c=target, s=6, alpha=0.3)
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.title('RECONSTRUCTION', fontsize=16, family='serif')

    #             plt.subplot(1, 4, 4)
    #             plt.semilogy(np.flip(np.sort(rms_sig)), '-ok')
    #             ground_truth = np.flip(np.sort(latent_rms_sig))
    #             ground_truth = latent_rms_sig
    #             plt.semilogy(scale_ground_truth(
    #                 ground_truth, rms_sig), '-ok', alpha=0.3)
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.title('SPECTRUM', fontsize=16, family='serif')

    #             plt.tight_layout()
    #             fig_idx = 4*dim_order + 2 * \
    #                 max(dim1_factor, 0) + max(dim2_factor, 0)
    #             plt.savefig(os.path.join(model.save_dir, 'figures', 'reconstruction_{}_{}.png'.format(epoch, fig_idx)))
    #             plt.close()


def scale_ground_truth(y, x):
    logy = (np.log(y)-np.min(np.log(y))) * \
        (np.max(np.log(x))-np.min(np.log(x)))
    logy /= np.max(np.log(y))-np.min(np.log(y))
    logy += np.min(np.log(x))
    return np.exp(logy)


def emnist_plot_samples(model, n_rows, dims_to_sample=torch.arange(784), temp=1):
    """
    Plots sampled digits. Each row contains all 10 digits with a consistent style
    """
    model.eval()
    fig = plt.figure(figsize=(10, n_rows))
    n_dims_to_sample = len(dims_to_sample)
    style_sample = torch.zeros(n_rows, 784)
    style_sample[:, dims_to_sample] = torch.randn(
        n_rows, n_dims_to_sample)*temp
    style_sample = style_sample.to(model.device)
    # style sample: (n_rows, n_dims)
    # mu,sig: (n_classes, n_dims)
    # latent: (n_rows, n_classes, n_dims)
    latent = style_sample.unsqueeze(
        1)*model.sig.unsqueeze(0) + model.mu.unsqueeze(0)
    latent.detach_()
    # data: (n_rows, n_classes, 28, 28)
    data = (model(latent.view(-1, 784), rev=True)
            [0]).detach().cpu().numpy().reshape(n_rows, 10, 28, 28)
    im = data.transpose(0, 2, 1, 3).reshape(n_rows*28, 10*28)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(model.save_dir, 'figures',
                f'epoch_{model.epoch+1:03d}', 'samples.png'), bbox_inches='tight', pad_inches=0.5)
    plt.close()


def emnist_plot_variation_along_dims(model, dims_to_plot):
    """
    Makes a plot for each of the given latent space dimensions. Each column contains all 10 digits
    with a consistent style. Each row shows the effect of varying the latent space value of the 
    chosen dimension from -2 to +2 standard deviations while keeping the latent space
    values of all other dimensions constant at the mean value. The rightmost column shows a heatmap
    of the absolute pixel difference between the column corresponding to -1 std and +1 std
    """
    os.makedirs(os.path.join(model.save_dir, 'figures',
                f'epoch_{model.epoch+1:03d}', 'variation_plots'))
    max_std = 2
    n_cols = 9
    model.eval()
    for i, dim in enumerate(dims_to_plot):
        fig = plt.figure(figsize=(n_cols+1, 10))
        style = torch.zeros(n_cols, 784)
        style[:, dim] = torch.linspace(-max_std, max_std, n_cols)
        style = style.to(model.device)
        # style: (n_cols, n_dims)
        # mu,sig: (n_classes, n_dims)
        # latent: (n_classes, n_cols, n_dims)
        latent = style.unsqueeze(
            0)*model.sig.unsqueeze(1) + model.mu.unsqueeze(1)
        latent.detach_()
        data = (model(latent.view(-1, 784), rev=True)
                [0]).detach().cpu().numpy().reshape(10, n_cols, 28, 28)
        im = data.transpose(0, 2, 1, 3).reshape(10*28, n_cols*28)
        # images at +1 and -1 std
        im_p1 = im[:, 28*2:28*3]
        im_m1 = im[:, 28*6:28*7]
        # full image with spacing between the two parts
        im = np.concatenate(
            [im, np.ones((10*28, 3)), np.abs(im_p1-im_m1)], axis=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots', f'variable_{i+1:03d}.png'),
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()


def emnist_plot_spectrum(model, sig_rms):
    fig = plt.figure(figsize=(12, 6))
    plt.semilogy(np.flip(np.sort(sig_rms)), 'k')
    plt.xlabel('Latent dimension (sorted)')
    plt.ylabel('Standard deviation (RMS across classes)')
    plt.title('Spectrum on EMNIST')
    plt.savefig(os.path.join(model.save_dir, 'figures',
                f'epoch_{model.epoch+1:03d}', 'spectrum.png'))
    plt.close()


def gaussian_plot(model, latent, data, target):
    """
    This function plots 8 figures of a reconstructed latent space, each for a different orientation of the 
    reconstructed latent space.
    """
    model.eval()
    model.cpu()
    x_axis = latent[:, 1]
    x_axis = (x_axis - x_axis.min()) / (x_axis.max() - x_axis.min())
    z_reconstructed = model(data).detach()
    sig = torch.stack([z_reconstructed[target == i].std(
        0, unbiased=False) for i in range(model.n_classes)])
    rms_sig = np.sqrt(np.mean(sig.numpy()**2, 0))
    latent_sig = torch.stack([latent[target == i].std(
        0, unbiased=False) for i in range(model.n_classes)])
    latent_rms_sig = np.sqrt(np.mean(latent_sig.numpy()**2, 0))

    for dim_order in range(2):
        for dim1_factor in [1, -1]:
            for dim2_factor in [1, -1]:
                fig = plt.figure(figsize=(12, 3.5))

                plt.subplot(1, 4, 1)
                plt.scatter(latent[:, 0], latent[:, 1],
                            c=x_axis, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('GROUND TRUTH', fontsize=16, family='serif')

                plt.subplot(1, 4, 2)
                plt.scatter(data[:, 0], data[:, 1], c=x_axis, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('OBSERVED DATA\n(PROJECTION)',
                          fontsize=16, family='serif')

                plt.subplot(1, 4, 3)
                dim1 = np.flip(np.argsort(rms_sig))[dim_order]
                dim2 = np.flip(np.argsort(rms_sig))[(1+dim_order) % 2]
                plt.scatter(
                    dim1_factor*z_reconstructed[:, dim1], dim2_factor*z_reconstructed[:, dim2], c=x_axis, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('RECONSTRUCTION', fontsize=16, family='serif')

                plt.subplot(1, 4, 4)
                plt.semilogy(np.flip(np.sort(rms_sig)), '-ok')
                ground_truth = np.flip(np.sort(latent_rms_sig))
                plt.semilogy(scale_ground_truth(
                    ground_truth, rms_sig), '-ok', alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('SPECTRUM', fontsize=16, family='serif')

                plt.tight_layout()
                fig_idx = 4*dim_order + 2 * \
                    max(dim1_factor, 0) + max(dim2_factor, 0)
                plt.savefig(os.path.join(model.save_dir, 'figures',
                            f'reconstruction_{fig_idx:d}.png'))
                plt.close()


def triangle_plot_variation_along_dims(model, dims_to_plot):
    """
    Makes a plot for each of the given latent space dimensions. Each column contains all 10 digits
    with a consistent style. Each row shows the effect of varying the latent space value of the 
    chosen dimension from -2 to +2 standard deviations while keeping the latent space
    values of all other dimensions constant at the mean value. The rightmost column shows a heatmap
    of the absolute pixel difference between the column corresponding to -1 std and +1 std
    """
    os.makedirs(os.path.join(model.save_dir, 'figures',
                f'epoch_{model.epoch+1:03d}', 'variation_plots'))
    max_std = 2
    n_cols = 9
    width = model.width
    n_pixel = width * width
    n_classes = model.n_classes
    model.eval()
    for i, dim in enumerate(dims_to_plot):
        fig = plt.figure(figsize=(n_cols+1, n_classes))
        style = torch.zeros(n_cols, n_pixel)
        style[:, dim] = torch.linspace(-max_std, max_std, n_cols)
        style = style.to(model.device)
        # style: (n_cols, n_dims)
        # mu,sig: (n_classes, n_dims)
        # latent: (n_classes, n_cols, n_dims)
        latent = style.unsqueeze(
            0)*model.sig.unsqueeze(1) + model.mu.unsqueeze(1)
        latent.detach_()
        data = model(latent.view(-1, n_pixel), rev=True).detach(
        ).cpu().numpy().reshape(n_classes, n_cols, width, width)
        im = data.transpose(0, 2, 1, 3).reshape(n_classes*width, n_cols*width)
        # images at +1 and -1 std
        im_p1 = im[:, width*2:width*3]
        im_m1 = im[:, width*6:width*7]
        # full image with spacing between the two parts
        im = np.concatenate(
            [im, np.ones((n_classes*width, 3)), np.abs(im_p1-im_m1)], axis=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots', f'variable_{i+1:03d}.png'),
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()


def triangle_plot_variation_along_dims_2s(model, dims_to_plot):
    """
    Makes a plot for each of the given latent space dimensions. Each column contains all 10 digits
    with a consistent style. Each row shows the effect of varying the latent space value of the 
    chosen dimension from -2 to +2 standard deviations while keeping the latent space
    values of all other dimensions constant at the mean value. The rightmost column shows a heatmap
    of the absolute pixel difference between the column corresponding to -1 std and +1 std
    """
    os.makedirs(os.path.join(model.save_dir, 'figures',
                f'epoch_{model.epoch+1:03d}', 'variation_plots'))
    max_std = 2
    n_cols = 9
    width = model.width
    n_pixel = width * width
    n_classes = model.n_classes
    model.eval()
    for i, dim in enumerate(dims_to_plot):
        fig = plt.figure(figsize=(n_cols+1, n_classes))
        style = torch.zeros(n_cols, n_pixel)
        style[:, dim] = torch.linspace(-max_std, max_std, n_cols)
        style = style.to(model.device)
        # style: (n_cols, n_dims)
        # mu,sig: (n_classes, n_dims)
        # latent: (n_classes, n_cols, n_dims)
        latent = style.unsqueeze(
            0)*model.sig.unsqueeze(1) + model.mu.unsqueeze(1)
        latent.detach_()
        data = model.net(model.net_2s(latent.view(-1, n_pixel), rev=True),
                         rev=True).detach().cpu().numpy().reshape(n_classes, n_cols, width, width)
        im = data.transpose(0, 2, 1, 3).reshape(n_classes*width, n_cols*width)
        # images at +1 and -1 std
        im_p1 = im[:, width*2:width*3]
        im_m1 = im[:, width*6:width*7]
        # full image with spacing between the two parts
        im = np.concatenate(
            [im, np.ones((n_classes*width, 3)), np.abs(im_p1-im_m1)], axis=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots', f'variable_{i+1:03d}.png'),
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()


def plot_scatter_along_dims(model, z, target, dims_to_plot):
    fig = plt.figure()
    ax = p3d.Axes3D(fig)
    ax.scatter(z[:, dims_to_plot[0]], z[:, dims_to_plot[1]],
               z[:, dims_to_plot[2]], c=target, alpha=0.5)
    plt.savefig(os.path.join(model.save_dir, 'figures',
                f'epoch_{model.epoch+1:03d}', f'scatter.png'))
