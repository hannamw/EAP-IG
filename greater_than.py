#%%
from pathlib import Path 

import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from greater_than_dataset import get_prob_diff, YearDataset, get_valid_years
from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

from attribute_vectorized import attribute_vectorized 
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

ds = YearDataset(get_valid_years(model.tokenizer, 1100, 1800), 1000, Path("potential_nouns.txt"), model.tokenizer)

clean = list(batch(ds.good_sentences, 9))
labels = list(batch(ds.years_YY, 9))
corrupted = list(batch(ds.bad_sentences, 9))

prob_diff = get_prob_diff(model.tokenizer)

# %%
# Instantiate a graph with a model
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: prob_diff(logits, labels), aggregation='sum')
scores = g.scores(sort=True)
#%%
# Apply a threshold
g.apply_threshold(scores[-120], absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=True)
print(g.count_included_edges())
gz = g.to_graphviz()
gz.draw(f'graph_gt_{model_name_noslash}_sum.png', prog='dot')
# %%
results = evaluate_graph(model, g, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))
print(results.mean())
# %%
# Instantiate a graph with a model
g2 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g2, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: prob_diff(logits, labels), aggregation='l2')
scores2 = g2.scores(sort=True)
#%%
# Apply a threshold
g2.apply_threshold(scores2[-120], absolute=False)
g2.prune_dead_nodes(prune_childless=True, prune_parentless=True)
print(g2.count_included_edges())
gz2 = g2.to_graphviz()
gz2.draw(f'graph_gt_{model_name_noslash}_l2.png', prog='dot')
# %%
results2 = evaluate_graph(model, g2, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))
print(results2.mean())
# %%
# Instantiate a graph with a model
g3 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g3, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: prob_diff(logits, labels), integrated_gradients=50)
scores3 = g3.scores(sort=True)
#%%
# Apply a threshold
g3.apply_threshold(scores3[-120], absolute=False)
g3.prune_dead_nodes(prune_childless=True, prune_parentless=True)
print(g3.count_included_edges())
gz3 = g3.to_graphviz()
gz3.draw(f'graph_gt_{model_name_noslash}_ig.png', prog='dot')
# %%
results3 = evaluate_graph(model, g3, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))
#%%
g_pareto = []
for i in range(50, 501, 50):
    g.apply_threshold(scores[-i], absolute=False)
    g.prune_dead_nodes(prune_childless=True, prune_parentless=True)
    n_edges = g.count_included_edges().item()
    results_i = evaluate_graph(model, g, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))
    g_pareto.append((n_edges, results_i.mean().item()))

g_pareto_x, g_pareto_y = list(zip(*g_pareto))

# %%
g3_pareto = []
for i in range(50, 501, 50):
    g3.apply_threshold(scores3[-i], absolute=False)
    g3.prune_dead_nodes(prune_childless=True, prune_parentless=True)
    n_edges = g3.count_included_edges().item()
    results3_i = evaluate_graph(model, g3, clean, corrupted, labels, lambda logits,corrupted_logits,input_lengths,labels: -prob_diff(logits, labels))
    g3_pareto.append((n_edges, results3_i.mean().item()))

g3_pareto_x, g3_pareto_y = list(zip(*g3_pareto))
# %%
fig, ax = plt.subplots()
ax.plot(g_pareto_x, g_pareto_y)
ax.plot(g3_pareto_x, g3_pareto_y)
ax.legend(['EAP', 'EAP-IG'])
ax.set_xlabel('Edges included (/32491)')
ax.set_ylabel('Loss recovery')
ax.set_title('EAP vs. EAP-IG')
# %%
