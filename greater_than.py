#%%
from pathlib import Path 

import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from greater_than_dataset import get_prob_diff, YearDataset, get_valid_years
from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode
from attribute import attribute_vectorized

from attribute_vectorized import attribute_vectorized as ave, get_activations
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

clean = list(batch(ds.good_sentences, 9))
labels = list(batch(ds.years_YY, 9))
corrupted = list(batch(ds.bad_sentences, 9))

input_length = 1 + len(model.tokenizer(ds.good_sentences[0])[0])
prob_diff = get_prob_diff(model.tokenizer)

# %%
# Instantiate a graph with a model and the length of the data
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute_vectorized(model, g, clean, corrupted, labels, prob_diff)
# Apply a threshold
g.apply_threshold(0.011, absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=False)
gz = g.to_graphviz()
gz.draw('graph_vectorized.png', prog='dot')

# %%
# Instantiate a graph with a model and the length of the data
g2 = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
a,b,c,d = ave(model, g2, clean, corrupted, labels, prob_diff)
# Apply a threshold
g2.apply_threshold(0.011, absolute=False)
g2.prune_dead_nodes(prune_childless=True, prune_parentless=False)
gz2 = g2.to_graphviz()
gz2.draw('graph_vectorized2.png', prog='dot')
# %%
input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients = get_activations(model, g2, clean[0], corrupted[0], prob_diff, labels[0])
# %%


#%%
g = Graph.from_model(model)
metric = prob_diff
graph = g
corrupted_inputs = corrupted[0]
clean_inputs = clean[0]
batch_size = len(clean_inputs)
n_pos = 1 + len(model.tokenizer(clean_inputs[0])[0])

input_activations_clean = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')
parent_activations_clean = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')

input_activations_corrupted = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')
parent_activations_corrupted = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')

child_gradients = torch.zeros((batch_size, n_pos, model.cfg.n_layers + 1, model.cfg.d_model), device='cuda')
attn_child_gradients = torch.zeros((batch_size, n_pos, n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv'), model.cfg.d_model), device='cuda')

processed_nodes = set()
processed_attn_layers = set()
fwd_hooks_clean = []
fwd_hooks_corrupted = []
bwd_hooks = []

def make_hook(t:torch.Tensor, index, unsqueeze=False):
    def hook_fn(activations, hook):
        acts = activations.detach()
        if unsqueeze:
            acts = acts.unsqueeze(2)
        t[index] = acts
    return hook_fn

for name, node in graph.nodes.items():
    name_non_positional = name if graph.n_pos is None else '_'.join(name.split('_')[:-1])
    if name_non_positional in processed_nodes:
        continue
    processed_nodes.add(name_non_positional)
    if isinstance(node, InputNode):
        fwd_hooks_clean.append((node.out_hook, make_hook(input_activations_clean, slice(None))))
        fwd_hooks_corrupted.append((node.out_hook, make_hook(input_activations_corrupted, slice(None))))
    elif isinstance(node, LogitNode):
        bwd_hooks.append((node.in_hook, make_hook(child_gradients, (slice(None), slice(None), model.cfg.n_layers))))
    elif isinstance(node, MLPNode):
        fwd_hooks_clean.append((node.out_hook, make_hook(parent_activations_clean, (slice(None), slice(None), node.layer, model.cfg.n_heads))))
        fwd_hooks_corrupted.append((node.out_hook, make_hook(parent_activations_corrupted, (slice(None), slice(None), node.layer, model.cfg.n_heads))))
        bwd_hooks.append((node.in_hook, make_hook(child_gradients, (slice(None), slice(None), node.layer))))
    elif isinstance(node, AttentionNode):
        if node.layer in processed_attn_layers:
            continue 
        processed_attn_layers.add(node.layer)
        fwd_hooks_clean.append((node.out_hook, make_hook(parent_activations_clean, (slice(None), slice(None), node.layer, slice(0, model.cfg.n_layers)))))
        fwd_hooks_corrupted.append((node.out_hook, make_hook(parent_activations_corrupted, (slice(None), slice(None), node.layer, slice(0, model.cfg.n_layers)))))
        for i, letter in enumerate('qkv'):
            letter_input_hook = f'blocks.{node.layer}.hook_{letter}_input' 
            bwd_hooks.append((letter_input_hook, make_hook(attn_child_gradients, (slice(None), slice(None), slice(None), node.layer, slice(None), i), unsqueeze=True)))
    else:
        raise ValueError(f"Invalid node: {node} of type {type(node)}")
#%%
with model.hooks(fwd_hooks=fwd_hooks_corrupted):
    _ = model(corrupted_inputs)

with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
    logits = model(clean_inputs)
    metric_value = metric(logits, labels)
    metric_value.backward()

    input_activation_differences = input_activations_corrupted - input_activations_clean
    parent_activation_differences = parent_activations_corrupted - parent_activations_clean
