from torch_geometric.data import Data

class PairData(Data):
    def __init__(self, edge_index_s=None, h_s=None, x_s=None, n_nodes_s=None,edge_index_t=None, h_t=None, x_t=None, n_nodes_t=None, y=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.h_s = h_s
        self.x_s = x_s
        self.n_nodes_s = n_nodes_s

        self.edge_index_t = edge_index_t
        self.h_t = h_t
        self.x_t = x_t
        self.n_nodes_t = n_nodes_t

        self.y = y