import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x_s, h_s, node_mask_s, edge_mask_s, x_t, h_t, node_mask_t, edge_mask_t, context):
    bs, n_nodes, n_dims = x_s.size()


    if args.probabilistic_model == 'diffusion':
        edge_mask_s = edge_mask_s.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x_s, node_mask_s)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x_s, h_s, x_t, h_t, node_mask_s=node_mask_s, edge_mask_s=edge_mask_s, node_mask_t=node_mask_t, edge_mask_t=edge_mask_t, context=context)

        N = node_mask_s.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z
