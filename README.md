# In-context RAG

## Environment Setup
+ Step 0: Create a new Conda virtual environment
```bash
conda create -n labelrag python==3.12 -c conda-forge -y
conda activate labelrag
```
+ Step 1: Install PyTorch 2.4 or [other versions](https://pytorch.org/)
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```
+ Step 2: Install dependencies
```bash
pip install transformers peft numpy pandas tqdm nest_asyncio huggingface_hub sentence_transformers
pip install deepspeed trl tensorboard loguru triton bitsandbytes tiktoken modelscope ogb
pip install flash-attn
```
+ Step 3: Install the corresponding version of `vllm` (compatible with PyTorch 2.4)
```bash
pip install vllm==0.6.3.post1
```
+ Step 4: Install [swift](https://github.com/modelscope/ms-swift), version 3.x is required
```bash
pip install 'ms-swift[llm]'
```
+ Step 5: Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
```bash
pip install torch_geometric
```
+ Step 6: Install [DGL](https://www.dgl.ai/pages/start.html) (only for loading graph data)
+ Step 7 (optional): Install FAISS
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
# pip install faiss-gpu
```
+ Step 8 (optional): Install `git-lfs` for downloading large models
```bash
apt-get install git-lfs
```

## Download Datasets
Download CSTAG-related datasets. The `ogbn-arxiv` dataset will be downloaded automatically in the code.
```bash
huggingface-cli download --repo-type dataset --resume-download Sherirto/CSTAG --local-dir CSTAG --local-dir-use-symlinks False
```
If the download fails in regions with restricted access, use a mirror source:
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --repo-type dataset --resume-download Sherirto/CSTAG --local-dir CSTAG --local-dir-use-symlinks False
```

## Download Models from ModelScope
```bash
git lfs clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct.git
git lfs clone https://www.modelscope.cn/qwen/Qwen2.5-7B-Instruct.git
git lfs clone https://www.modelscope.cn/LLM-Research/gemma-2-9b-it.git
git lfs clone https://www.modelscope.cn/LLM-Research/Mistral-7B-Instruct-v0.3.git
git lfs clone https://www.modelscope.cn/LLM-Research/Phi-3.5-mini-instruct.git
huggingface-cli download --resume-download sentence-transformers/all-mpnet-base-v2 --local-dir ./all-mpnet-base-v2
```
**Note**: After cloning with `git lfs`, the `.git` folder in the model directory may take up significant space. Make sure to remove it if necessary:
```bash
cd Meta-Llama-3.1-8B-Instruct && rm -rf .git
```

## Run the Code
+ Preprocess datasets and construct prompts:
```python
python generate_prompt_node.py
```
+ Configure the `gpu_ids` in the shell script, then run all models in one command (multi-GPU inference with native PyTorch):
```bash
bash scripts/run_node.sh 
```

For more parameter details, refer to the Swift documentation: [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)
