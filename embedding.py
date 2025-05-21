from dataset import load_tg_dataset 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import os.path as osp
import torch

device = torch.device('cuda')
model_path = './all-mpnet-base-v2'    
model = SentenceTransformer(model_path, device=device, 
                            trust_remote_code=True, 
                            model_kwargs=dict(torch_dtype=torch.bfloat16))
def encode(prompts):
    normalize_embeddings = True
    batch_size = 32
    emb = model.encode(prompts, convert_to_tensor=True, show_progress_bar=True, 
                       normalize_embeddings=normalize_embeddings, batch_size=batch_size).cpu()
    
    print(emb.shape)
    return emb

os.makedirs('DTGB/temp', exist_ok =True)

# for name in (['Enron', 'GDELT', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_ubuntu', 'Amazon_movies', 'Yelp']):
for name in (['Enron', 'ICEWS1819', 'Stack_elec', 'Stack_ubuntu', 'Amazon_movies', 'Yelp']):
    print(name)
    data = load_tg_dataset(name)
    os.makedirs(osp.join('temp', name), exist_ok =True)

    embedding_path = osp.join('DTGB/temp', f'{name}/{name}_x.pt')
    if not osp.exists(embedding_path):
        prompts = data['entity_text']['text'].values[1:]
        assert len(prompts) == data['num_nodes']
        emb = encode(prompts)
        torch.save(emb, embedding_path)
    
    embedding_path = osp.join('DTGB/temp', f'{name}/{name}_msg.pt')
    if not osp.exists(embedding_path):
        prompts = data['relation_text']['text'].values[1:]
        assert len(prompts) == data['num_relations']
        emb = encode(prompts)
        torch.save(emb, embedding_path)