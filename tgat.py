import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
from lastgl.data import TemporalData
from sklearn.preprocessing import LabelEncoder
from dataset import load_tg_dataset 

from lastgl.datasets import JODIEDataset
from lastgl.loader import EventLoader
from lastgl.models import TGAT
from lastgl.transforms import TemporalSplit
from lastgl.utils import Measure


def main():

    # name = 'Enron'
    name = 'Yelp'
    data = load_tg_dataset(name)
    edge_df = data['edge_list']
    src = torch.tensor(edge_df.u.values).long()
    dst = torch.tensor(edge_df.i.values).long()
    t = torch.tensor(edge_df.ts.values).float()
    r = torch.tensor(edge_df.r.values).long()-1
    y = torch.tensor(LabelEncoder().fit_transform(edge_df.label.values)).long()
    train_mask = torch.tensor(edge_df.split.values=='train')
    val_mask = torch.tensor(edge_df.split.values=='val')
    test_mask = torch.tensor(edge_df.split.values=='test')

    # test_idx = torch.where(test_mask)[:10000]
    # test_mask = torch.zeros_like(val_mask)
    # test_mask[test_idx] = True
    data = TemporalData(src=src-1, dst=dst-1, t=t, y=y, num_nodes=data['num_nodes'],
                        train_mask=train_mask, val_mask=val_mask,
                        test_mask=test_mask)
    
    assert data.src.min() == 0
    # assert data.dst.min() == 0
    assert data.y.min() == 0
    assert r.min() == 0
    
    # data.x = torch.zeros(data.num_nodes, 1)  # assign zero node features
    data.x = torch.load(f'DTGB/temp/{name}/{name}_x.pt').float()
    # data.msg = torch.zeros(src.size(0), 1)  # assign zero edge features
    data.msg = torch.load(f'DTGB/temp/{name}/{name}_msg.pt').float()[r]

    train_loader = EventLoader(data, num_neighbors=[20, 20],
                               input_events=data.train_mask, num_workers=8,
                               batch_size=256, shuffle=True)
    valid_loader = EventLoader(data, num_neighbors=[20, 20],
                               input_events=data.val_mask, num_workers=8,
                               batch_size=256)
    test_loader = EventLoader(data, num_neighbors=[20, 20],
                              input_events=data.test_mask, num_workers=8,
                              batch_size=256)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TGAT(
        in_channels=data.x.size(-1),
        out_channels=data.y.max().item() + 1,
        hidden_channels=128,
        edge_dim=data.msg.size(-1),
        heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    measure = Measure('acc')
    loss_fn = torch.nn.CrossEntropyLoss()

    def train(loader):
        model.train()
        pbar = tqdm(loader)
        total_loss = 0
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.t, batch.msg)
            src = batch.src[:batch.batch_size]
            out = out[src].squeeze()
            label = batch.y[:batch.batch_size]
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_description(f'Train Loss = {loss.item():.4f}')

        return total_loss / len(train_loader)

    @torch.no_grad()
    def test(loader):
        preds = []
        labels = []
        model.eval()
        for batch in tqdm(loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.t, batch.msg)
            src = batch.src[:batch.batch_size]
            out = out[src].squeeze()
            label = batch.y[:batch.batch_size]
            preds.append(out)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        return measure(labels, preds)

    for epoch in range(1, 11):

        loss = train(train_loader)
        val_auc = test(valid_loader)
        test_auc = test(test_loader)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Val ACC: {val_auc:.2%}, Test ACC: {test_auc:.2%}')


if __name__ == "__main__":
    main()
