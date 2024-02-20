#%%
from pathlib import Path 

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

from attribute_mem import attribute
from evaluate_graph import evaluate_graph, evaluate_baseline
from utils import batch, kl_div, inflow_outflow_difference
#%%
model_name = 'gpt2-medium'
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
#%%
df: pd.DataFrame = pd.read_csv('../data/ioi/ioi_dataset.csv')
df = df.sample(frac=1)
df = df.head(500)

batch_size = 8
clean = list(batch(df['clean'], batch_size))
corrupted = list(batch(df['corrupted'], batch_size))

labels = [torch.tensor(x) for x in batch(list(zip(df['correct_idx'], df['incorrect_idx'])), batch_size)]

def prob_diff(clean_logits, corrupted_logits, input_length, labels, mean=True, logit=False):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if logit:
        good_bad_probs = torch.gather(clean_logits, -1, labels.to(clean_probs.device))
    else:
        good_bad_probs = torch.gather(clean_probs, -1, labels.to(clean_probs.device))

    # remember it's reversed to make it a loss
    results = -(good_bad_probs[:, 0] - good_bad_probs[:, 1])
    return results.mean() if mean else results
# %%
baseline = evaluate_baseline(model, clean, corrupted, labels, lambda *args:  -1 * prob_diff(*args, mean=False, logit=True))
print(baseline.mean())
#%%
# Instantiate a graph with a model
g = Graph.from_model(model)
# Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
attribute(model, g, clean, corrupted, labels, lambda *args: prob_diff(*args, logit=True), integrated_gradients=30)
#%%
inflow_outflow_difference(g)
#%%
# Apply a threshold
g.apply_greedy(400, absolute=False)
g.prune_dead_nodes(prune_childless=True, prune_parentless=True)
gz = g.to_graphviz()
gz.draw(f'images/{model_name_noslash}/ioi.png', prog='dot')
performance = evaluate_graph(model, g, clean, corrupted, labels, lambda *args:  -1 * prob_diff(*args, mean=False, logit=True))
print(performance.mean())
g.to_json(f'graphs/{model_name_noslash}/ioi.json')
# %%
