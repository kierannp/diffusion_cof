import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None):
    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
        #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), molsxsample=None, context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    x_ts = [s[0] for s in molsxsample]
    h_ts = [s[1] for s in molsxsample]
    edge_index_ts = [s[2] for s in molsxsample]
    n_nodes_ts = [s[3] for s in molsxsample]
    batch_max_t_n_nodes = max(n_nodes_ts)
    max_edges = max([ei.size(1) for ei in edge_index_ts])

    x_t = torch.zeros((batch_size, batch_max_t_n_nodes,3))
    h_t = torch.zeros((batch_size, batch_max_t_n_nodes,5))
    edge_index_t = torch.zeros((2, max_edges*batch_size))

    for d in range(batch_size):
        x_t[d,:x_ts[d].size(0)] = x_ts[d]
        h_t[d,:x_ts[d].size(0)] = h_ts[d]
        edge_index_t[0, d*max_edges:d*max_edges+edge_index_ts[d].size(1)] = edge_index_ts[d][0]
        edge_index_t[1, d*max_edges:d*max_edges+edge_index_ts[d].size(1)] = edge_index_ts[d][1]

    node_mask_s = torch.zeros(batch_size, max_n_nodes)
    node_mask_t = torch.zeros(batch_size, batch_max_t_n_nodes)
    for i in range(batch_size):
        node_mask_s[i, 0:nodesxsample[i]] = 1
        node_mask_t[i, 0:x_ts[i].size(0)] = 1

    # Compute edge_mask
    edge_mask_s = node_mask_s.unsqueeze(1) * node_mask_s.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask_s.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask_s *= diag_mask
    edge_mask_s = edge_mask_s.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask_s = node_mask_s.unsqueeze(2).to(device)

    edge_mask_t = node_mask_t.unsqueeze(1) * node_mask_t.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask_t.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask_t *= diag_mask
    edge_mask_t = edge_mask_t.view(batch_size * batch_max_t_n_nodes * batch_max_t_n_nodes, 1).to(device)
    node_mask_t = node_mask_t.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask_s
    else:
        context = None

    h_t = {'categorical':h_t}

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask_s, edge_mask_s, x_t, h_t, node_mask_t, edge_mask_t, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask_s)
        assert_mean_zero_with_mask(x, node_mask_s)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask_s)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask_s)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask_s


def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=9, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, molsxsample=molsxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask

def get_adj_matrix(n_nodes, batch_size, device):
    edges_dict = {}
    if n_nodes in edges_dict:
        edges_dic_b = edges_dict[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)
            edges = [torch.LongTensor(rows).to(device),
                        torch.LongTensor(cols).to(device)]
            edges_dic_b[batch_size] = edges
            return edges
    else:
        edges_dict[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)