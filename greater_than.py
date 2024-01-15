#%%
from pathlib import Path 

import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from greater_than_dataset import get_prob_diff, YearDataset, get_valid_years
from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode
from attribute import attribute, attribute_vectorized, get_activations

#%%
model_name = 'EleutherAI/pythia-160m'
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
clean_short = ds.good_sentences[:12]
labels_short = ds.years_YY[:12]
corrupted_short = ds.bad_sentences[:12]
clean = list(batch(ds.good_sentences, 10))
labels = list(batch(ds.years_YY, 10))
corrupted = list(batch(ds.bad_sentences, 10))

#%%
input_length = 1 + len(model.tokenizer(ds.good_sentences[0])[0])

prob_diff = get_prob_diff(model.tokenizer)
# %%
g = Graph.from_model_positional(model, input_length)
attribute(model, g, clean_short, corrupted_short, prob_diff, labels_short)
g.apply_threshold(0.0005, False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=True)
gz = g.to_graphviz()
gz.draw('graph.png', prog='dot')

#%%
# clean = list(batch(ds.good_sentences[:12], 4))
# labels = list(batch(ds.years_YY[:12], 4))
# corrupted = list(batch(ds.bad_sentences[:12], 4))
g2 = Graph.from_model_positional(model, input_length)
attribute_vectorized(model, g2, clean, corrupted, prob_diff, labels)
g2.apply_threshold(0.0005, False)
g2.prune_dead_nodes(prune_childless=True, prune_parentless=True)
gz2 = g2.to_graphviz()
gz2.draw('graph_vectorized.png', prog='dot')

# %%
