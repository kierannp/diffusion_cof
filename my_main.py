import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import glob
import os
import time
import matplotlib.pyplot as plt
import sys
import copy
import torch
import time
import pickle
sys.path.insert(1, 'e3_diffusion_for_molecules')
import utils
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
from qm9.utils import prepare_context, compute_mean_mad
import qm9.visualizer as vis
from qm9.analyze import check_stability
from qm9.sampling import sample_chain, sample
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import qm9.utils as qm9_utils
from qm9 import losses
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import shutil
from datetime import datetime
import torch
import json
import mbuild as mb
import gmso
from gmso.external.convert_networkx import to_networkx
import networkx as nx
from urllib.request import urlopen
from urllib.parse import quote
from torch.distributions.categorical import Categorical
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from sklearn.metrics import mean_squared_error
from typing import List, Optional, Union
import torch.utils.data

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import my_models
import tribology_dataset
import my_losses
import my_sample
import my_analyze

def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    def make_edge_index(atom_mask, bs, n_nodes):
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1).to(device)
        return edge_mask

    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x_s, atom_mask_s = tg.utils.to_dense_batch(data.x_s, data.x_s_batch)
        x_t, atom_mask_t = tg.utils.to_dense_batch(data.x_t, data.x_t_batch)
        batch_size_s, n_nodes_s, _ = x_s.size()
        batch_size_t, n_nodes_t, _ = x_t.size()
        x_s, atom_mask_s = x_s.to(device, dtype), atom_mask_s.to(device, dtype)
        x_t, atom_mask_t = x_t.to(device, dtype), atom_mask_t.to(device, dtype)
        # atom_positions_s = dense_positions_s.view(batch_size_s * n_nodes_s, -1).to(device, dtype)
        # atom_positions_t = dense_positions_t.view(batch_size_t * n_nodes_t, -1).to(device, dtype)

        edge_mask_s = make_edge_index(atom_mask_s, batch_size_s, n_nodes_s)
        edge_mask_s = edge_mask_s.to(device, dtype)

        edge_mask_t = make_edge_index(atom_mask_t, batch_size_t, n_nodes_t)
        edge_mask_t = edge_mask_t.to(device, dtype)

        atom_mask_s = atom_mask_s.unsqueeze(2)
        atom_mask_t = atom_mask_t.unsqueeze(2)

        h_s, h_s_mask = tg.utils.to_dense_batch(data.h_s, data.h_s_batch)
        h_t, h_t_mask = tg.utils.to_dense_batch(data.h_t, data.h_t_batch)
        h_s, h_t, h_s_mask, h_t_mask = h_s.to(device, dtype), h_t.to(device, dtype), h_s_mask.to(device, dtype), h_t_mask.to(device, dtype)
        # one_hot_s = one_hot_s.view(batch_size_s * n_nodes_s, -1).to(device)
        # one_hot_t = one_hot_t.view(batch_size_t * n_nodes_t, -1).to(device)
        # edges_s = qm9_utils.get_adj_matrix(n_nodes_s, batch_size_s, device)
        # edges_t = qm9_utils.get_adj_matrix(n_nodes_t, batch_size_t, device)
        label = data.y.to(device, dtype)

        # x = data['positions'].to(device, dtype)
        # node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        # edge_mask = data['edge_mask'].to(device, dtype)
        # one_hot = data['one_hot'].to(device, dtype)
        # x = remove_mean_with_mask(x, node_mask)

        x_s = remove_mean_with_mask(x_s, atom_mask_s)
        x_t = remove_mean_with_mask(x_t, atom_mask_t)

        if args.data_augmentation:
            x_s = utils.random_rotation(x_s).detach()

        # check_mask_correct([x_s, one_hot_s], atom_mask_s)
        # check_mask_correct([x_t, one_hot_t], atom_mask_t)
        # assert_mean_zero_with_mask(x_s, atom_mask_s)
        # assert_mean_zero_with_mask(x_t, atom_mask_t)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        h_s = {'categorical': h_s, 'integer': charges}
        h_t = {'categorical': h_t, 'integer': charges}

        if len(args.conditioning) > 0:
            context = prepare_context(['COF'], data, property_norms).to(device, dtype)
            assert_correctly_masked(context, atom_mask_s)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = my_losses.compute_loss_and_nll(args, model, nodes_dist,
                                                                  x_s, h_s, atom_mask_s, edge_mask_s,
                                                                  x_t, h_t, atom_mask_t, edge_mask_t, 
                                                                  context)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            # if len(args.conditioning) > 0:
            #     save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            # # save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
            # #                       batch_id=str(i))  
            # # sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
            # #                                 prop_dist, epoch=epoch)
            # print(f'Sampling took {time.time() - start:.2f} seconds')

            # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=False)
            # vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=False)
            # if len(args.conditioning) > 0:
            #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
            #                         wandb=False, mode='conditional')
        # wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break

def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    def make_edge_index(atom_mask, bs, n_nodes):
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1).to(device)
        return edge_mask
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        n_iterations = len(loader)
        for i, data in enumerate(loader):
            x_s, atom_mask_s = tg.utils.to_dense_batch(data.x_s, data.x_s_batch)
            x_t, atom_mask_t = tg.utils.to_dense_batch(data.x_t, data.x_t_batch)
            batch_size_s, n_nodes_s, _ = x_s.size()
            batch_size_t, n_nodes_t, _ = x_t.size()
            x_s, atom_mask_s = x_s.to(device, dtype), atom_mask_s.to(device, dtype)
            x_t, atom_mask_t = x_t.to(device, dtype), atom_mask_t.to(device, dtype)

            edge_mask_s = make_edge_index(atom_mask_s, batch_size_s, n_nodes_s)
            edge_mask_s = edge_mask_s.to(device, dtype)

            edge_mask_t = make_edge_index(atom_mask_t, batch_size_t, n_nodes_t)
            edge_mask_t = edge_mask_t.to(device, dtype)

            atom_mask_s = atom_mask_s.unsqueeze(2)
            atom_mask_t = atom_mask_t.unsqueeze(2)

            h_s, h_s_mask = tg.utils.to_dense_batch(data.h_s, data.h_s_batch)
            h_t, h_t_mask = tg.utils.to_dense_batch(data.h_t, data.h_t_batch)
            h_s, h_t, h_s_mask, h_t_mask = h_s.to(device, dtype), h_t.to(device, dtype), h_s_mask.to(device, dtype), h_t_mask.to(device, dtype)
            label = data.y.to(device, dtype)

            x_s = remove_mean_with_mask(x_s, atom_mask_s)
            x_t = remove_mean_with_mask(x_t, atom_mask_t)

            if args.data_augmentation:
                x_s = utils.random_rotation(x_s).detach()

            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            h_s = {'categorical': h_s, 'integer': charges}
            h_t = {'categorical': h_t, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context(['COF'], data, property_norms).to(device, dtype)
                assert_correctly_masked(context, atom_mask_s)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = my_losses.compute_loss_and_nll(args, model, nodes_dist,
                                                                  x_s, h_s, atom_mask_s, edge_mask_s,
                                                                  x_t, h_t, atom_mask_t, edge_mask_t, 
                                                                  context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size_s
            n_samples += batch_size_s
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples
    
def generate_node_histogram(dataset):
    node_hist = {}
    for d in dataset:
        sn = int(d['n_nodes_s']) 
        if sn in node_hist.keys():
            node_hist[sn] += 1
        else:
            node_hist[sn] = 1
    return node_hist
def generate_mol_histogram(dataset):
    mol_hist = {}
    for d in dataset:
        mol_info = (d['x_t'], d['h_t'], d['edge_index_t'], d['n_nodes_t'])
        if mol_info in mol_hist.keys():
            mol_hist[mol_info] += 1
        else:
            mol_hist[mol_info] = 1
    return mol_hist
def compute_mean_mad_from_dataset(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = torch.Tensor([float(d['y']) for d in dataloader])
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms
def prepare_context(conditioning, minibatch, property_norms):
    x_s, atom_mask_s = tg.utils.to_dense_batch(minibatch.x_s, minibatch.x_s_batch)
    x_t, atom_mask_t = tg.utils.to_dense_batch(minibatch.x_t, minibatch.x_t_batch)
    batch_size_s, n_nodes_s, _ = x_s.size()
    batch_size_t, n_nodes_t, _ = x_t.size()
    node_mask_s = atom_mask_s.unsqueeze(2)
    node_mask_t = atom_mask_t.unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch['y']
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # Global feature.
            assert properties.size() == (batch_size_s,)
            reshaped = properties.view(batch_size_s, 1, 1).repeat(1, n_nodes_s, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        # elif len(properties.size()) == 2 or len(properties.size()) == 3:
        #     # Node feature.
        #     assert properties.size()[:2] == (batch_size, n_nodes)

        #     context_key = properties

        #     # Inflate if necessary.
        #     if len(properties.size()) == 2:
        #         context_key = context_key.unsqueeze(2)

        #     context_list.append(context_key)
        #     context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask_s
    assert context.size(2) == context_node_nf
    return context
def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x

def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')

def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        molsxsample = mol_dist.sample(batch_size)
        one_hot, charges, x, node_mask = my_sample.sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, molsxsample=molsxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = my_analyze.analyze_stability_for_molecules(molecules, dataset_info, dataset.elements)

    # wandb.log(validity_dict)
    # if rdkit_tuple is not None:
    #     wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict

def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = my_sample.sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x

args = edict({'conditioning': ['alpha'], 'include_charges': False, 'condition_time':True,
                'context_node_nf': 1, 'nf': 192, 'n_layers': 4, 'batch_size':32, 'num_workers':2,
                'attention': True, 'tanh': True, 'model': 'egnn_dynamics', 'norm_constant': 1,
                'inv_sublayers':1, 'sin_embedding':False, 'normalization_factor': 1, 'aggregation_method': 'sum',
                'probabilistic_model': 'diffusion', 'diffusion_steps': 1000, 'diffusion_noise_schedule':'polynomial_2',
                'diffusion_noise_precision':1e-5, 'diffusion_loss_type':'l2', 'normalize_factors': [1,8,1], 
                'lr':1e-4, 'start_epoch':0, 'n_epochs':10, 'ema_decay':.9999, 'augment_noise':0,
                'data_augmentation':False, 'ode_regularization': 1e-3, 'clip_grad':True, 'dataset':'qm9', 
                'filter_n_atoms':None, 'datadir':'/Users/kieran/structure-encoding/e3_diffusion_for_molecules/qm9/data/qm9', 
                'property':'alpha','remove_h':True, 'resume':None, 'no_wandb':True, 'no_cuda':True, 'exp_name':'knp_testing',
                'wandb_usr':'kieran', 'lr':1e-4, 'n_stability_samples':5, 'dequantization':'deterministic', 'dp':False,
                'n_report_steps':5, 'test_epochs':1, 'visualize_every_batch':1, 'break_train_epoch':False, 'cuda':False,
                'save_model':False})
try:
    shutil.rmtree('/Users/kieran/diffusion_cof/processed')
except:
    print('Dataset has not been processed')

conditioned = True
dataset = tribology_dataset.TribologyDataset('.')
dataset.shuffle()

dataset_info = {}
batch_size = 32
dtype = torch.float32
in_node_nf = len(dataset.element2vec)
device = torch.device("cuda" if args.cuda else "cpu")

start, mid = int(len(dataset)*.4), int(len(dataset)*.9)
train_dataset = dataset[:start]
test_dataset = dataset[start:mid]
valid_dataset = dataset[mid:]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['h_s', 'h_t', 'x_s', 'x_t'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['h_s', 'h_t', 'x_s', 'x_t'])
valid_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['h_s', 'h_t', 'x_s', 'x_t'])

n_nodes_hist = generate_node_histogram(dataset)
mol_hist = generate_mol_histogram(dataset)
dataset_info['max_n_nodes'] = max(n_nodes_hist.keys())

node_dist = my_models.DistributionNodes(n_nodes_hist)
mol_dist = my_models.DistributionMolecules(mol_hist)
prop_dist = my_models.DistributionProperty(dataset, ['COF'])

mini_b = next(iter(train_loader))

if len(args.conditioning) > 0:
    property_norms = compute_mean_mad_from_dataset(dataset, ['COF'])
    context_dummy = prepare_context(['COF'], mini_b, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

if args.condition_time:
    dynamics_in_node_nf = in_node_nf + 1
else:
    print('Warning: dynamics model is _not_ conditioned on time.')
    dynamics_in_node_nf = in_node_nf

net_dynamics = my_models.Cond_EGNN_dynamics(
    in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
    n_dims=3, device=device, hidden_nf=args.nf,
    act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
    attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
    inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
    normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

if args.probabilistic_model == 'diffusion':
    vdm = my_models.CondEnVariationalDiffusion(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges
        )

if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = vdm.to(device)
optim = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4, amsgrad=True,
    weight_decay=1e-12)
print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


########################################### main  ###############################################
if args.resume is not None:
    flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
    optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
    model.load_state_dict(flow_state_dict)
    optim.load_state_dict(optim_state_dict)

# Initialize dataparallel if enabled and possible.
if args.dp and torch.cuda.device_count() > 1:
    print(f'Training using {torch.cuda.device_count()} GPUs')
    model_dp = torch.nn.DataParallel(model.cpu())
    model_dp = model_dp.cuda()
else:
    model_dp = model

# Initialize model copy for exponential moving average of params.
if args.ema_decay > 0:
    model_ema = copy.deepcopy(model)
    ema = flow_utils.EMA(args.ema_decay)

    if args.dp and torch.cuda.device_count() > 1:
        model_ema_dp = torch.nn.DataParallel(model_ema)
    else:
        model_ema_dp = model_ema
else:
    ema = None
    model_ema = model
    model_ema_dp = model_dp

best_nll_val = 1e8
best_nll_test = 1e8
for epoch in range(args.start_epoch, args.n_epochs):
    start_epoch = time.time()
    train_epoch(args=args, loader=train_loader, epoch=epoch, model=model, model_dp=model_dp,
                model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                nodes_dist=node_dist, dataset_info=dataset_info,
                gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
    print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

    if epoch % args.test_epochs == 0:
        # if isinstance(model, en_diffusion.EnVariationalDiffusion):
        #     # wandb.log(model.log_info(), commit=True)

        if not args.break_train_epoch:
            analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=node_dist,
                                dataset_info=dataset_info, device=device,
                                prop_dist=prop_dist, n_samples=args.n_stability_samples)
        nll_val = test(args=args, loader=valid_loader, epoch=epoch, eval_model=model_ema_dp,
                        partition='Val', device=device, dtype=dtype, nodes_dist=node_dist,
                        property_norms=property_norms)
        nll_test = test(args=args, loader=test_loader, epoch=epoch, eval_model=model_ema_dp,
                        partition='Test', device=device, dtype=dtype,
                        nodes_dist=node_dist, property_norms=property_norms)

        if nll_val < best_nll_val:
            best_nll_val = nll_val
            best_nll_test = nll_test
            if args.save_model:
                args.current_epoch = epoch + 1
                utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                    pickle.dump(args, f)

            if args.save_model:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
        print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
        print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
        # wandb.log({"Val loss ": nll_val}, commit=True)
        # wandb.log({"Test loss ": nll_test}, commit=True)
        # wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)