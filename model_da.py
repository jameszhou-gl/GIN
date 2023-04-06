import torch
import torch.nn as nn
import numpy as np
from time import time
import datetime
import os
from collections import OrderedDict

from data import make_dataloader, make_dataloader_emnist
from plot import artificial_data_reconstruction_plot, da_synthetic_data_reconstruction_plot, emnist_plot_samples, emnist_plot_spectrum, emnist_plot_variation_along_dims

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from metrics import *


class GIN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dataset = args.dataset
        self.n_epochs = args.n_epochs
        self.epochs_per_line = args.epochs_per_line
        self.lr = args.lr
        self.lr_schedule = args.lr_schedule
        self.batch_size = args.batch_size
        self.save_frequency = args.save_frequency
        self.mcc_frequency = args.mcc_frequency
        self.incompressible_flow = bool(args.incompressible_flow)
        self.empirical_vars = bool(args.empirical_vars)
        self.init_identity = bool(args.init_identity)
        self.dim_c = args.dim_c
        self.dim_s = args.dim_s

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.timestamp = str(int(time()))
        now = datetime.datetime.now()
        self.save_dir = os.path.join(
            './domain_adaptation_{}_save/'.format(self.dataset), now.strftime("%Y-%m%d-%H-%M-%S"))

        if self.dataset == 'synthetic':
            os.makedirs(self.save_dir)
            os.makedirs(os.path.join(self.save_dir, 'model_save'))
            os.makedirs(os.path.join(self.save_dir, 'figures'))
            data_dir = os.path.join(self.save_dir, 'data')
            os.makedirs(data_dir)
            self.n_dims = 10
            self.net = construct_net_da(
                coupling_block='gin' if self.incompressible_flow else 'glow', n_dims=self.n_dims, init_identity=args.init_identity)
            assert type(args.n_domains) is int
            self.n_domains = args.n_domains
            self.log_file = open(os.path.join(self.save_dir, 'log.txt'), 'a')
            self.log_file.write(str(args)+'\n')
            
            if args.load_existing_dataset:
                print('\n load files from {}\n'.format(args.load_existing_dataset))
                self.log_file.write('\n load files from {}\n'.format(args.load_existing_dataset))
                self.latent = torch.load(os.path.join(
                    args.load_existing_dataset, 'latent.pt'))
                self.data = torch.load(os.path.join(
                    args.load_existing_dataset, 'data.pt'))
                self.target = torch.load(os.path.join(
                    args.load_existing_dataset, 'labels.pt'))
            else:
                self.latent, self.data, self.target = generate_artificial_data_domain_adaptation(
                    n_domains=self.n_domains, n_data_points=args.n_data_points, n_dims=self.n_dims, dim_s=self.dim_s, dim_c=self.dim_c, data_dir = data_dir)
            self.train_loader = make_dataloader(
                self.data, self.target, self.batch_size)



        else:
            raise RuntimeError("Check dataset name. Doesn't match.")

        if not self.empirical_vars:
            self.mu = nn.Parameter(torch.zeros(
                self.n_domains, self.n_dims).to(self.device)).requires_grad_()
            self.log_sig = nn.Parameter(torch.zeros(
                self.n_domains, self.n_dims).to(self.device)).requires_grad_()
            # initialize these parameters to reasonable values
            self.set_mu_sig(init=True)

        self.to(self.device)

    def forward(self, x, rev=False):
        # generative process with rev=False, z -> x
        x, logdet_J = self.net(x, rev=rev)
        return x, logdet_J

    def train_model(self):
        # with open(os.path.join(self.save_dir, 'log.txt'), 'w') as f:
        self.log_file.write(
            f'incompressible_flow {self.incompressible_flow}\n')
        self.log_file.write(f'empirical_vars {self.empirical_vars}\n')
        self.log_file.write(f'init_identity {self.init_identity}\n')
        print(f'\nTraining model for {self.n_epochs} epochs \n')
        self.log_file.write(f'\nTraining model for {self.n_epochs} epochs \n')

        self.train()
        self.to(self.device)
        print('  time     epoch    iteration         loss       last checkpoint')
        self.log_file.write(
            '  time     epoch    iteration         loss       last checkpoint \n')
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_schedule)
        losses = []
        t0 = time()
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.empirical_vars:
                    # first check that std will be well defined
                    if min([sum(target == i).item() for i in range(self.n_domains)]) < 2:
                        # don't calculate loss and update weights -- it will give nan or error
                        # go to next batch
                        continue
                optimizer.zero_grad()
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                z, logdet_J = self.net(data)          # latent space variable
                # check all values in logdet_j are 0
                assert torch.all(
                    logdet_J == 0), "log of Jacobian determinant must be zero"
                if self.empirical_vars:
                    # we only need to calculate the std
                    sig = torch.stack(
                        [z[target == i].std(0, unbiased=False) for i in range(self.n_domains)])
                    # negative log-likelihood for gaussian in latent space
                    loss = 0.5 + sig[target].log().mean(1) + \
                        0.5*np.log(2*np.pi)
                else:
                    m = self.mu[target]
                    ls = self.log_sig[target]
                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(
                        0.5*(z-m)**2 * torch.exp(-2*ls) + ls, 1) + 0.5*np.log(2*np.pi)
                # with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
                #     f.write(f'logdet_J: {logdet_J}\n')
                loss -= logdet_J / self.n_dims
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
                self.make_plots(epoch)
                self.train()
                self.to(self.device)
            if (epoch+1) % self.mcc_frequency == 0:
                mcc = self.compute_mcc()
                print('epoch: {}; \r mcc score is {} \n'.format(epoch, mcc))
                self.log_file.write(
                    'epoch: {}; \r mcc score is {} \n'.format(epoch, mcc))
                self.train()
                self.to(self.device)

        self.log_file.close()

    def print_loss(self, loss, batch_idx, epoch, t0, new_line=False):
        n_batches = len(self.train_loader)
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{self.n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
            self.log_file.write(print_str+' '*40)
            self.log_file.write('\n')
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

    def make_plots(self, epoch):
        if self.dataset == 'synthetic':
            da_synthetic_data_reconstruction_plot(
                self, self.latent, self.data, self.target, epoch)
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")

    def compute_mcc(self):
        self.eval()
        self.cpu()
        z_rec = self(self.data)[0]
        sig = torch.stack([z_rec[self.target == i].std(
            0, unbiased=False) for i in range(self.n_domains)])
        rms_sig = np.sqrt(np.mean(sig.detach().numpy()**2, 0))
        dim_z_c = np.flip(np.argsort(rms_sig))[:self.dim_c]

        z_rec = torch.cat([z_rec[self.target == i]
                          for i in range(self.n_domains)])
        z_c = z_rec[:, dim_z_c.copy()]
        latent = torch.cat([self.latent[self.target == i]
                           for i in range(self.n_domains)])
        latent_c = latent[:, self.dim_s: self.dim_s+self.dim_c]
        
        return mean_corr_coef_np(z_c.detach().numpy(), latent_c.detach().numpy())

        # return mean_corr_coef_pt(z_c, latent_c)

    def set_mu_sig(self, init=False, n_batches=40):
        if self.empirical_vars or init:
            examples = iter(self.test_loader)
            n_batches = min(n_batches, len(examples))
            latent = []
            target = []
            for _ in range(n_batches):
                data, targ = next(examples)
                data += torch.randn_like(data)*1e-2
                self.eval()
                latent.append((self(data.to(self.device))[0]).detach().cpu())
                target.append(targ)
            latent = torch.cat(latent, 0)
            target = torch.cat(target, 0)
        if self.empirical_vars:
            self.mu = torch.stack([latent[target == i].mean(0)
                                  for i in range(10)]).to(self.device)
            self.sig = torch.stack([latent[target == i].std(0)
                                   for i in range(10)]).to(self.device)
        else:
            if init:
                self.mu.data = torch.stack(
                    [latent[target == i].mean(0) for i in range(10)])
                self.log_sig.data = torch.stack(
                    [latent[target == i].std(0) for i in range(10)]).log()
            else:
                self.sig = self.log_sig.exp().detach()


def subnet_fc_10d(c_in, c_out, init_identity):
    subnet = nn.Sequential(nn.Linear(c_in, 10), nn.ReLU(),
                           nn.Linear(10, 10), nn.ReLU(),
                           nn.Linear(10,  c_out))
    if init_identity:
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_net_10d(coupling_block, init_identity=True):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(10, name='input')]

    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor': lambda c_in, c_out: subnet_fc_10d(
                                 c_in, c_out, init_identity), 'clamp': 2.0},
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                             {'seed': np.random.randint(2**31)},
                             name=F'permute_{k+1}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)


def construct_net_da(coupling_block, n_dims, init_identity=True):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(n_dims, name='input')]

    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor': lambda c_in, c_out: subnet_fc_10d(
                                 c_in, c_out, init_identity), 'clamp': 2.0},
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                             {'seed': np.random.randint(2**31)},
                             name=F'permute_{k+1}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)


def subnet_fc(c_in, c_out):
    width = 392
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv1(c_in, c_out):
    width = 16
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv2(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_net_emnist(coupling_block):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(1, 28, 28, name='input')]
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


def construct_net_triangle(coupling_block):
    # todo
    pass


# function is here rather than in data.py to prevent circular import
def generate_artificial_data_10d(n_clusters, n_data_points):
    latent_means = torch.rand(n_clusters, 2)*10 - 5         # in range (-5, 5)
    latent_stds = torch.rand(n_clusters, 2)*2.5 + 0.5      # in range (0.5, 3)

    labels = torch.randint(n_clusters, size=(n_data_points,))
    latent = latent_means[labels] + \
        torch.randn(n_data_points, 2)*latent_stds[labels]
    latent = torch.cat([latent, torch.randn(n_data_points, 8)*1e-2], 1)

    random_transf = construct_net_10d('glow', init_identity=False)
    data = random_transf(latent)[0].detach()

    return latent, data, labels

# function is here rather than in data.py to prevent circular import


def generate_artificial_data_domain_adaptation(n_domains, n_data_points, n_dims=10, dim_s=1, dim_c=2, data_dir=None):
    mu_u = torch.rand(n_domains, dim_s)*10 - 4  # in range (-4, 4)
    sig_u = torch.rand(n_domains, dim_s)*0.99 + 0.01  # in range (-0.01, 1)
    labels = torch.randint(n_domains, size=(n_data_points,))
    z_s = mu_u[labels] + \
        torch.randn(n_data_points, dim_s)*sig_u[labels]
    z_c = torch.randn(n_data_points, dim_c)
    z_noise = torch.randn(n_data_points, n_dims-dim_s-dim_c)*1e-2
    latent = torch.cat([z_s, z_c, z_noise], 1)

    random_transf = construct_net_da(
        'glow', n_dims=n_dims, init_identity=False)
    data = random_transf(latent)[0].detach()
    torch.save(latent, os.path.join(data_dir, 'latent.pt'))
    torch.save(data, os.path.join(data_dir, 'data.pt'))
    torch.save(labels, os.path.join(data_dir, 'labels.pt'))

    return latent, data, labels
