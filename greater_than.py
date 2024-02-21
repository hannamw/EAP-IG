#%%
from pathlib import Path 

import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from greater_than_dataset import get_prob_diff, YearDataset, get_valid_years
from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode
from evaluate_graph import evaluate_graph


from attribute import attribute
from attribute_mem import attribute as attribute_mem
#%%
model_name = 'gpt2-small'
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name, device='cuda')
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

N = 1000
batch_size = 12

ds = YearDataset(get_valid_years(model.tokenizer, 1100, 1800), N, Path("potential_nouns.txt"), model.tokenizer)

clean = list(batch(ds.good_sentences, batch_size))
labels = list(batch(ds.years_YY, batch_size))
corrupted = list(batch(ds.bad_sentences, batch_size))

prob_diff = get_prob_diff(model.tokenizer)

# %%
# Instantiate a graph with a model
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)

attribute(model, g, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: prob_diff(logits, labels),)
scores = g.scores(sort=True)
#%%
g.apply_greedy(400, absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=True)
print(g.count_included_edges())
gz = g.to_graphviz()
gz.draw(f'graph_gt_{model_name_noslash}.png', prog='dot')
results = evaluate_graph(model, g, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))

# %%
# Instantiate a graph with a model
g2 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_mem(model, g2, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: prob_diff(logits, labels),)
scores2 = g2.scores(sort=True)

#%%
g2.apply_greedy(400, absolute=False)
g2.prune_dead_nodes(prune_childless=True, prune_parentless=True)
gz2 = g2.to_graphviz()
gz2.draw(f'graph_gt_{model_name_noslash}.png', prog='dot')
results2 = evaluate_graph(model, g2, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))
