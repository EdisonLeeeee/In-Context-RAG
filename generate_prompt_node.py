import os
import torch
import numpy as np
from tqdm import tqdm
from swift.utils import write_to_jsonl, read_from_jsonl
from dataset import load_dataset
from sentence_transformers import SentenceTransformer

def build_index(x):
    import faiss
    dimension = x.shape[-1]
    # index = faiss.IndexFlatIP(dimension)
    # index = faiss.IndexFlatL2(dimension)
    res = faiss.StandardGpuResources()  # use GPU
    index = faiss.GpuIndexFlatL2(res, dimension)    
    # index = faiss.GpuIndexFlatIP(res, dimension)    
    index.add(x.float().cpu())
    return index
    
def search(index, x, k=20):
    distance, topk = index.search(x.float().cpu(), k)
    return distance, topk
    
topk = 3  # the number of used neighbors for each node
temp_dir = 'node_prompts'
os.makedirs(temp_dir, exist_ok=True)


raw_input_template = """Classify the following text into one of the predefined categories: {category}. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

Category:"""
for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)
    data_list = []
    for t, y in zip(text['text'], labels):
        query = raw_input_template.format(category=category, text=t)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(f'{temp_dir}/{dataset}_raw.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')

print(query)

rag_template = """Classify the following text into one of the predefined categories: {category}. Use the provided references to enhance your understanding of the topic and context for accurate classification. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

References: 
{references}

Category:"""

# huggingface-cli download --resume-download sentence-transformers/all-mpnet-base-v2 --local-dir ./all-mpnet-base-v2
model_path = './all-mpnet-base-v2'
model = SentenceTransformer(model_path, device='cuda:0',
                            trust_remote_code=True,
                            model_kwargs=dict(torch_dtype=torch.bfloat16))

for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    neighbor = text['neighbour']
    txt = text['text']
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)

    embedding = model.encode(txt, convert_to_tensor=True, show_progress_bar=True,
                             normalize_embeddings=True, batch_size=32).cpu()
    # retrieval_x = data.x
    retrieval_x = embedding
    index = build_index(retrieval_x)
    _, retrieved_topk = search(index, retrieval_x, topk+1)    
    retrieved_topk = retrieved_topk[:, 1:]

    data_list = []
    for t, nbr, y in zip(txt, retrieved_topk, labels):
        references = ''
        for i, s in enumerate(txt[nbr]):
            references += f'{i+1}. {s}\n'
        references = references.rstrip()
        query = rag_template.format(
            category=category, text=t, references=references)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(f'{temp_dir}/{dataset}_rag_{topk}.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')
print(query)


query_rag_template = """Classify the following text into one of the predefined categories: {category}. Use the provided references to enhance your understanding of the topic and context for accurate classification. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

References: 
{references}

Category:"""
for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    neighbor = text['neighbour']
    txt = text['text']
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)

    data_list = []
    for t, nbr, y in zip(txt, neighbor, labels):
        references = ''
        nbr = eval(nbr)[:topk]
        for i, s in enumerate(txt[nbr]):
            references += f'{i+1}. {s}\n'
        references = references.rstrip()
        query = query_rag_template.format(
            category=category, text=t, references=references)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(f'{temp_dir}/{dataset}_query_rag_{topk}.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')
print(query)


label_rag_template = """Classify the following text into one of the predefined categories: {category}. Use the provided references and corresponding categories to enhance your understanding of the topic and context for accurate classification. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

References: 
{references}

Category:"""
for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    neighbor = text['neighbour']
    txt = text['text']
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)

    data_list = []
    for t, nbr, y in zip(txt, neighbor, labels):
        references = ''
        nbr = eval(nbr)[:topk]
        for i, n in enumerate(nbr):
            s = txt[n]
            references += f'{i+1}. {s}\nCategory: {labels[n]}\n'
        references = references.rstrip()
        query = label_rag_template.format(
            category=category, text=t, references=references)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(f'{temp_dir}/{dataset}_label_rag_{topk}.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')
print(query)

label_only_rag_template = """Classify the following text into one of the predefined categories: {category}. Use the provided categories of reference texts to enhance your understanding of the topic and context for accurate classification. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

References: 
{references}

Category:"""
for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    neighbor = text['neighbour']
    txt = text['text']
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)

    data_list = []
    for t, nbr, y in zip(txt, neighbor, labels):
        references = ''
        nbr = eval(nbr)[:topk]
        for i, n in enumerate(nbr):
            s = txt[n]
            references += f'{i+1}. Category: {labels[n]}\n'
        references = references.rstrip()
        query = label_only_rag_template.format(
            category=category, text=t, references=references)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(
        f'{temp_dir}/{dataset}_label_only_rag_{topk}.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')
print(query)


few_shot_template = """Classify the following text into one of the predefined categories: {category}. Use the provided few-shot examples to enhance your understanding of the topic and context for accurate classification. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

Examples: 
{references}

Category:"""
for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    txt = text['text']
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)

    data_list = []
    for t, y in zip(txt, labels):
        references = ''
        few_shot_example_id = np.random.choice(np.arange(len(data)), topk).tolist()
        for i, n in enumerate(few_shot_example_id):
            s = txt[n]
            references += f'{i+1}. Text: {s}\nCategory: {labels[n]}\n'
        references = references.rstrip()
        query = few_shot_template.format(
            category=category, text=t, references=references)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(f'{temp_dir}/{dataset}_few_shot_{topk}.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')
print(query)


one_shot_template = """Classify the following text into one of the predefined categories: {category}. Use the provided few-shot examples to enhance your understanding of the topic and context for accurate classification. Make your decision based on the main topic and overall content of the text. If the text is ambiguous or does not clearly fit into any category, choose the closest match. Provide only the category name as the output.

Text: {text}

Examples: 
{references}

Category:"""
for dataset in tqdm(['Cora', 'Pubmed', 'History', 'Children', 'Photo', 'Computers', 'Arxiv', 'Fitness']):
    data, text = load_dataset(dataset, root='./CSTAG')
    txt = text['text']
    labels = text['category']
    category = labels.unique().tolist()
    category = ', '.join(category)

    data_list = []
    for t, y in zip(txt, labels):
        references = ''
        few_shot_example_id = np.random.choice(np.arange(len(data)), 1).tolist()
        for i, n in enumerate(few_shot_example_id):
            s = txt[n]
            references += f'{i+1}. Text: {s}\nCategory: {labels[n]}\n'
        references = references.rstrip()
        query = one_shot_template.format(
            category=category, text=t, references=references)
        data_list.append(dict(query=query, response=y))
    write_to_jsonl(f'{temp_dir}/{dataset}_few_shot_1.jsonl', data_list)
    print(f'{dataset} has {len(data_list)} samples.')
print(query)
