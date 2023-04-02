import torch
import torch.nn as nn
import numpy as np
from time import time
import os
from collections import OrderedDict

from data import make_dataloader, make_dataloader_triangle_2
from plot import gaussian_plot, artificial_data_reconstruction_plot, triangle_plot_variation_along_dims, plot_scatter_along_dims

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class GIN(nn.Module):
    def __init__(self, dataset, n_epochs, epochs_per_line, lr, lr_schedule, batch_size, save_frequency, incompressible_flow, empirical_vars, data_root_dir='./', n_data_points=None, init_identity=True):
        super().__init__()

        self.dataset = dataset
        self.n_epochs = n_epochs
        self.epochs_per_line = epochs_per_line
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.save_frequency = min(save_frequency, n_epochs)
        self.incompressible_flow = bool(incompressible_flow)
        self.empirical_vars = bool(empirical_vars)
        self.init_identity = bool(init_identity)

        self.device = 'cuda: 3' if torch.cuda.is_available() else 'cpu'
        self.timestamp = str(int(time()))

        if self.dataset == 'triangle':
            self.n_classes = 2
            self.width = 32
            self.n_dims = self.width * self.width
            self.net = construct_net_triangle(
                coupling_block='gin' if self.incompressible_flow else 'glow')
            self.save_dir = os.path.join('/home/guanglinzhou/code/cgm/GIN/triangle_save/', self.timestamp)
            self.train_loader = make_dataloader_triangle_2(
                self.batch_size, train=True, root_dir='/home/guanglinzhou/code/cgm/GIN/triangle')
            self.test_loader = make_dataloader_triangle_2(
                100, train=False, root_dir='/home/guanglinzhou/code/cgm/GIN/triangle')
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")

        if not empirical_vars:
            self.mu = nn.Parameter(torch.zeros(
                1, self.n_dims).to(self.device)).requires_grad_()
            self.log_sig = nn.Parameter(torch.zeros(
                1, self.n_dims).to(self.device)).requires_grad_()
            # initialize these parameters to reasonable values
            self.set_mu_sig(init=True)

        self.to(self.device)

    def forward(self, x, rev=False):
        x = self.net(x, rev=rev)
        return x

    def train_model(self):
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'log.txt'), 'w') as f:
            f.write(f'batch size {self.batch_size}\n')
            f.write(f'incompressible_flow {self.incompressible_flow}\n')
            f.write(f'empirical_vars {self.empirical_vars}\n')
            f.write(f'init_identity {self.init_identity}\n')
        os.makedirs(os.path.join(self.save_dir, 'model_save'))
        os.makedirs(os.path.join(self.save_dir, 'figures'))
        print(f'\nTraining model for {self.n_epochs} epochs \n')
        self.to(self.device)
        self.net.train()
        print('  time     epoch    iteration         loss       last checkpoint')
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_schedule)
        losses = []
        t0 = time()
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            for batch_idx, (data, target) in enumerate(self.train_loader):
                #target = target[:, 0].int()
                if self.empirical_vars:
                    # first check that std will be well defined
                    if min([sum(target == i).item() for i in range(self.n_classes)]) < 2:
                        # don't calculate loss and update weights -- it will give nan or error
                        # go to next batch
                        continue
                optimizer.zero_grad()
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                # logdet_j is 0
                z, logdet_J = self.net(data)          # latent space variable
                #logdet_J = self.net.log_jacobian(run_forward=False)
                if self.empirical_vars:
                    # we only need to calculate the std
                    #sig = torch.stack([z.std(0, unbiased=False)])
                    #sig_total = torch.stack([z.std(0, unbiased=False)])
                    # print('target.shape: {}, dtype of target: {}'.format( target.shape, target.dtype))
                    target = target.to(torch.long)
                    # print('target.shape: {}, dtype of target: {}'.format( target.shape, target.dtype))
                    mu = torch.stack([z[target == i].mean(0) for i in range(self.n_classes)])
                    sig = torch.stack(
                        [z[target == i].std(0, unbiased=False) for i in range(self.n_classes)])
                    # negative log-likelihood for gaussian in latent space
                    # + 0.1 * ((mu[0] - mu[1])**2).sum() + 0.5*np.log(2*np.pi)
                    loss = 0.5 + sig.log().mean()
                else:
                    m = self.mu[target]
                    ls = self.log_sig[target]
                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(
                        0.5*(z-m)**2 * torch.exp(-2*ls) + ls, 1) + 0.5*np.log(2*np.pi)
                #loss -= logdet_J / self.n_dims
                loss = loss.mean()
                self.print_loss(loss.item(), batch_idx, epoch, t0)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
            if (epoch+1) % self.epochs_per_line == 0:
                avg_loss = np.mean(losses)
                self.print_loss(avg_loss, batch_idx, epoch, t0, new_line=True)
                losses = []
            sched.step()
            if (epoch+1) % self.save_frequency == 0:
                self.save(os.path.join(self.save_dir,
                          'model_save', f'{epoch+1:03d}.pt'))
                self.make_plots()

    def print_loss(self, loss, batch_idx, epoch, t0, new_line=False):
        n_batches = len(self.train_loader)
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{self.n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
        else:
            last_save = (epoch//self.save_frequency)*self.save_frequency
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')

    def save(self, fname):
        state_dict = OrderedDict((k, v) for k, v in self.state_dict(
        ).items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)

    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['model'])

    def make_plots(self):
        if self.dataset == 'triangle':
            os.makedirs(os.path.join(self.save_dir, 'figures',
                        f'epoch_{self.epoch+1:03d}'))
            self.set_mu_sig()
            sig_rms = np.sqrt(
                np.mean((self.sig**2).detach().cpu().numpy(), axis=0))
            #emnist_plot_samples(self, n_rows=1)
            #emnist_plot_spectrum(self, sig_rms)
            n_dims_to_plot = 10
            top_sig_dims = np.flip(np.argsort(sig_rms))
            dims_to_plot = top_sig_dims[:n_dims_to_plot]
            triangle_plot_variation_along_dims(self, dims_to_plot)

            examples = iter(self.test_loader)
            latent = []
            target = []
            for _ in range(40):
                data, targ = next(examples)
                self.to(self.device)
                self.eval()
                # latent.append(self(data.to(self.device)).detach().cpu())
                latent.append(data)
                target.append(targ)
            latent = torch.cat(latent[:40], 0)
            target = torch.cat(target[:40], 0)
            plot_scatter_along_dims(self, latent, target, dims_to_plot)
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")

    def set_mu_sig(self, init=False, n_batches=40):
        if self.empirical_vars or init:
            examples = iter(self.test_loader)
            n_batches = min(n_batches, len(examples))
            latent = []
            target = []
            for _ in range(n_batches):
                data, targ = next(examples)
                #data = data[targ==self.cond]
                #data += torch.randn_like(data)*1e-2
                # self.to(self.device)
                # self.eval()
                # latent.append(self(data.to(self.device)).detach().cpu())
                latent.append(data)
                target.append(targ)
            latent = torch.cat(latent[:n_batches], 0)
            target = torch.cat(target[:n_batches], 0)
        if self.empirical_vars:
            self.mu = torch.stack([latent[target == i].mean(0)
                                  for i in range(self.n_classes)]).to(self.device)
            self.sig = torch.stack([latent[target == i].std(0)
                                   for i in range(self.n_classes)]).to(self.device)
        else:
            if init:
                self.mu.data = torch.stack(
                    [latent[target == i].mean(0) for i in range(self.n_classes)])
                self.log_sig.data = torch.stack(
                    [latent[target == i].std(0) for i in range(self.n_classes)]).log()
            else:
                self.sig = self.log_sig.exp().detach()


def subnet_fc(c_in, c_out):
    width = 512
    act = nn.ReLU()
    subnet = nn.Sequential(nn.Linear(c_in, width), act,
                           nn.Linear(width, width), act,
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv1(c_in, c_out):
    width = 16
    act = nn.ReLU()
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), act,
                           nn.Conv2d(width, width, 3, padding=1), act,
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv2(c_in, c_out):
    width = 32
    act = nn.ReLU()
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), act,
                           nn.Conv2d(width, width, 3, padding=1), act,
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_net_triangle(coupling_block):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(1, 32, 32, name='input')]
    nodes.append(
        Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor': subnet_conv1, 'clamp': 2.0},
                             name=F'coupling_conv1_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed': np.random.randint(2**31)},
                             name=F'permute_conv1_{k}'))

    nodes.append(
        Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample2'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor': subnet_conv2, 'clamp': 2.0},
                             name=F'coupling_conv2_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed': np.random.randint(2**31)},
                             name=F'permute_conv2_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    for k in range(2):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                             name=F'coupling_fc_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed': np.random.randint(2**31)},
                             name=F'permute_fc_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)
