#%%
from pathlib import Path 

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

from attribute_vectorized import attribute_vectorized 
#%%
model_name = 'gpt2'
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
#%%
def batch(iterable, n:int=1):
   current_batch = []
   for item in iterable:
       current_batch.append(item)
       if len(current_batch) == n:
           yield current_batch
           current_batch = []
   if current_batch:
       yield current_batch

df: pd.DataFrame = pd.read_csv('../data/hypernymy/dataset.csv')
df = df.sample(frac=1)
df = df.head(100)


clean = list(batch(df['clean'], 9))
corrupted = list(batch(df['corrupted'], 9))

answer_tensors = [torch.tensor(eval(idxs)) for idxs in df['answers_idx']]
corrupted_answer_tensors = [torch.tensor(eval(idxs)) for idxs in df['corrupted_answers_idx']]
labels = list(batch(list(zip(answer_tensors, corrupted_answer_tensors)), 9))

def prob_diff(clean_logits, corrupted_logits, input_length, labels, mean=True):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = []
    for i, (ls,corrupted_ls) in enumerate(labels):
        clean_probs[i][ls.to(clean_probs.device)].sum() - clean_probs[i][corrupted_ls.to(clean_probs.device)].sum()
    results = torch.stack(results)
    return results.mean() if mean else results
# %%
# Instantiate a graph with a model
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g, clean, corrupted, labels, prob_diff)
# Apply a threshold
g.apply_threshold(0.011, absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=False)
gz = g.to_graphviz()
gz.draw('graph.png', prog='dot')