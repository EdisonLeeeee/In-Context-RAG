import torch
import dgl
import numpy as np
import pandas as pd
import os
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





def temporal_split(edge_df: pd.DataFrame,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15
                   ) -> pd.DataFrame:

    assert 0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0
    assert val_ratio + test_ratio < 1.0, "val_ratio + test_ratio must < 1"

    val_time, test_time = np.quantile(
        edge_df['ts'].values,
        [1. - val_ratio - test_ratio,
         1. - test_ratio]
    )

    def _mark(ts):
        if ts < val_time:
            return "train"
        elif ts < test_time:
            return "val"
        else:
            return "test"

    edge_df["split"] = edge_df["ts"].apply(_mark)
    return edge_df

CATEGORY_NAME = {
    # Amazon / Yelp / Google Map
    "Amazon_movies": {0: "Very bad", 1: "Bad", 2: "Moderate",
                      3: "Good", 4: "Very good"},
    "Yelp":          {0: "Very bad", 1: "Bad", 2: "Moderate",
                      3: "Good", 4: "Very good"},
    "Googlemap_CT":  {0: "Very bad", 1: "Bad", 2: "Moderate",
                      3: "Good", 4: "Very good"},

    # Enron
    "Enron": {0: "notes_inbox", 1: "discussion_threads", 2: "california",
              3: "personal", 4: "calendar", 5: "tw_commercial_group",
              6: "_americas", 7: "power", 8: "deal_communication",
              9: "gir"},

    # Stack Overflow email and ubuntu
    "Stack_elec":   {0: "Useless", 1: "Useful"},
    "Stack_ubuntu": {0: "Useless", 1: "Useful"},
}

def load_tg_dataset(name: str, base_dir: str = "DTGB", val_ratio=0.15, test_ratio=0.15):
    """

    Parameters
    -------
    name : str
        dataset name，e.g., 'Amazon_movies'
    base_dir : str, default 'DTGB'
        root dir

    Return
    -------
    data : dict
        {
            'edge_list'      : DataFrame，
            'entity_text'    : DataFrame，
            'relation_text'  : DataFrame，
            'num_nodes'      : int，
            'num_relations'  : int
        }
    """
    assert name in ['Amazon_movies', 'Enron', 'GDELT', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_ubuntu', 'Yelp']
    folder = os.path.join(base_dir, name)

    edge_path = os.path.join(folder, "edge_list.csv")
    edge_df   = pd.read_csv(edge_path)
    num_nodes = max(edge_df["u"].max(), edge_df["i"].max())
    num_rels  = edge_df["r"].max()

    ent_path  = os.path.join(folder, "entity_text.csv")
    ent_df    = pd.read_csv(ent_path)

    rel_path  = os.path.join(folder, "relation_text.csv")
    rel_df    = pd.read_csv(rel_path)

    # assert len(edge_df) == len(rel_df["text"]), (len(edge_df), len(rel_df["text"]))
    
    edge_df = temporal_split(edge_df, val_ratio, test_ratio)
    if name in ['GDELT', 'ICEWS1819']:
        edge_df["label"] = edge_df.apply(lambda x: rel_df.loc[x["r"], "text"], axis=1)
    elif name in ['Amazon_movies', 'Googlemap_CT', 'Yelp']:
        edge_df["label"] = (edge_df["label"]-1).map(CATEGORY_NAME[name])
    else:
        edge_df["label"] = edge_df["label"].map(CATEGORY_NAME[name])

    assert edge_df["label"].isna().sum() == 0
    return {
        "edge_list": edge_df,
        "entity_text": ent_df,
        "relation_text": rel_df,
        "num_nodes": num_nodes,
        "num_relations": num_rels
    }

# data = load_tg_dataset("Yelp")
# print("nodes:", data["num_nodes"], "| relations:", data["num_relations"])
# print(data["edge_list"].head())
