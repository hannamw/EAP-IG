#%%
from pathlib import Path 

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

from attribute_vectorized import attribute_vectorized 
from evaluate_graph import evaluate_graph, evaluate_kl
#%%
model_name = 'EleutherAI/pythia-160m'
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

df = pd.read_csv('clean_EventsAdapt_SentenceSet.csv')

#%%
clean = list(batch(df['sp'], 9))
corrupted = list(batch(df['si'], 9))
labels = list(batch(df['sp'], 9))

def kl_div(clean_logits, corrupted_logits, input_length, labels, mean=True, flipped=True):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    corrupted_logits = corrupted_logits[idx, input_length - 1]

    clean_logprobs = torch.log_softmax(clean_logits, dim=-1)
    corrupted_logprobs = torch.log_softmax(corrupted_logits, dim=-1)

    kl = F.kl_div(clean_logprobs, corrupted_logprobs, log_target=True, reduction='none') if flipped else F.kl_div(corrupted_logprobs, clean_logprobs, log_target=True, reduction='none')
    return kl.mean() if mean else kl
#%%
# Instantiate a graph with a model
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g, clean, corrupted, labels, lambda *args: -1 * kl_div(*args))

#%%
# Apply a threshold
g.apply_threshold(1.9e-8, absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=True)
gz = g.to_graphviz()
gz.draw(f'graph_{model_name_noslash}.png', prog='dot')
# %%
results = evaluate_graph(model, g, clean, corrupted, labels, lambda a,b,c,d: kl_div(a,b,c,d, flipped=False))
# %%
corrupted_results = evaluate_kl(model, corrupted, clean)
# %%
print(results)
print(corrupted_results)
# %%
