#%%
from pathlib import Path 

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

from attribute_vectorized import attribute_vectorized, get_npos_input_lengths
from evaluate_graph import evaluate_graph
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

df: pd.DataFrame = pd.read_csv('../data/fact-retrieval/country_capital_dataset.csv')
df = df.sample(frac=1)
# df = df.head(100)

batch_size = 8
clean = list(batch(df['clean'], 8))
corrupted = list(batch(df['corrupted'], 8))

country_tensors = [torch.tensor(idx) for idx in df['country_idx']]
corrupted_country_tensors = [torch.tensor(idx) for idx in df['corrupted_country_idx']]
labels = [torch.tensor(x) for x in batch(list(zip(df['country_idx'], df['corrupted_country_idx'])), 8)]

def prob_diff(clean_logits, corrupted_logits, input_length, labels, mean=True):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)

    good_bad_probs = torch.gather(clean_probs, -1, labels.to(clean_probs.device))

    # remember it's reversed to make it a loss
    results = -(good_bad_probs[:, 0] - good_bad_probs[:, 1])
    return results.mean() if mean else results
# %%
# Instantiate a graph with a model
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g, clean, corrupted, labels, prob_diff)
scores = g.scores(sort=True)
#%%
# Apply a threshold
g.apply_threshold(0.002, absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=False)
gz = g.to_graphviz()
gz.draw(f'graph_fact-retrieval_{model_name_noslash}.png', prog='dot')
#%%
performance = evaluate_graph(model, g, clean, corrupted, labels, lambda *args:  -1 * prob_diff(*args, mean=False))
print(performance.mean())
# %%
# Instantiate a graph with a model
g2 = Graph.from_model(model)

for node in g2.nodes.values():
    node.in_graph = True

for edge in g2.edges.values():
    edge.in_graph = True 
    edge.score = 0.002
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
baseline = evaluate_graph(model, g2, clean, corrupted, labels, lambda *args:  -1 * prob_diff(*args, mean=False))
print(baseline.mean())

# %%
