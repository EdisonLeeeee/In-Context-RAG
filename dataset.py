import torch
import dgl
import numpy as np
import pandas as pd
import os.path as osp
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from torch_geometric.utils import from_dgl, index_to_mask, to_scipy_sparse_matrix, to_undirected
from torch_geometric import seed_everything
from sklearn.preprocessing import LabelEncoder


def csr_to_neighbors_list(csr_adj):
    neighbors_list = []
    for node in range(csr_adj.shape[0]):
        start, end = csr_adj.indptr[node], csr_adj.indptr[node + 1]
        neighbors = csr_adj.indices[start:end]
        neighbors_list.append(str(list(neighbors)))
    return neighbors_list
    
def load_dataset(name, root='./CSTAG', seed=42, use_ori_feature=False, setting='node'):
    root = osp.expanduser(root)
    assert name in ['Fitness', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Cora', 'Pubmed'], name
    dir_path = osp.join(root, name)
    graph_path = osp.join(dir_path, f'{name}.pt')
    feature_path = osp.join(dir_path, 'Feature', f'{name}_roberta_base_512_cls.npy')
    table_path = osp.join(dir_path, f'{name}.csv')
    seed_everything(seed)

    if name == 'Arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-arxiv')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
        data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
        data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)
        text = pd.read_csv(table_path).rename(columns={'ID': 'node_id'})
        if not use_ori_feature:
            x = torch.from_numpy(np.load(feature_path)).float()
            data.x = x        

    elif name in ['Cora', 'Pubmed']:
        path = f'./tsgfm/{name.lower()}/processed/geometric_data_processed.pt'
        data = torch.load(path)[0]
        text = pd.DataFrame(dict(text=data.raw_texts, category=data.category_names, ID=np.arange(data.num_nodes)))
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
        x = torch.from_numpy(np.load(feature_path)).float()
        text = pd.read_csv(table_path)
        data = from_dgl(graph)
        data.x = x    
        data.y = data.pop('label', None)
        data = RandomNodeSplit(num_val=0.2, num_test=0.2)(data)
    
    data.edge_index = to_undirected(data.edge_index)
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    text['neighbour'] = csr_to_neighbors_list(adj)   
    
    if setting == 'node':
        print('Using node classification settings')
        return data, text
    else:
        print('Using Link prediction settings')
        train, val, test = RandomLinkSplit(num_val=0.05, num_test=0.1,
                        is_undirected=True,
                        split_labels=True,
                        add_negative_train_samples=False)(data)        

        return (train, val, test), text
